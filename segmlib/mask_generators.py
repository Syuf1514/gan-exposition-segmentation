import torch

from abc import ABC, abstractmethod
from torch import nn
from sklearn.mixture import GaussianMixture

from .unet import UNet


rgb_channels = 3
epsilon = 1e-3


class MaskGenerator(ABC, nn.Module):
    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.log_probs(images, shifted_images, k) for k in range(self.n_classes)], dim=1)
        return log_masks

    @abstractmethod
    def log_probs(self, images, shifted_images, k):
        pass

    @staticmethod
    def create(name, n_classes):
        if name == 'affine':
            return AffineMaskGenerator(n_classes)
        elif name == 'grayscale':
            return GrayscaleMaskGenerator(n_classes)
        elif name == 'neural':
            return NeuralMaskGenerator(n_classes)
        elif name == 'gaussian':
            return GaussianMaskGenerator(n_classes)
        else:
            raise ValueError(f'unknown mask generator "{name}"')


class AffineMaskGenerator(MaskGenerator):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.affine_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels) for _ in range(n_classes)])
        # self.inv_sigma_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels, bias=False)
        #                                           for _ in range(n_classes)])
        for operator in self.affine_operators:
            operator.weight.data = torch.eye(rgb_channels)
        # for operator in self.inv_sigma_operators:
        #     operator.weight.data = torch.eye(rgb_channels)
        self.register_parameter('log_sigma', nn.Parameter(torch.tensor(0.0)))

    def log_probs(self, images, shifted_images, k):
        # difference = self.inv_sigma_operators[k](shifted_images - self.affine_operators[k](images))
        difference = (shifted_images - self.affine_operators[k](images)) / self.log_sigma.exp()
        dependent_part = -0.5 * torch.einsum('bijc, bijc -> bij', difference, difference)
        # constant_part = torch.slogdet(self.inv_sigma_operators[k].weight).logabsdet
        constant_part = -rgb_channels * self.log_sigma
        return dependent_part + constant_part


class GrayscaleMaskGenerator(MaskGenerator):
    def __init__(self, n_classes):
        super().__init__()
        if n_classes != 2:
            raise ValueError('grayscale mask generator only supports binary segmentation')
        self.n_classes = n_classes

        self.weight = nn.Linear(rgb_channels, 1, bias=False)
        self.weight.weight.data = torch.tensor([0.299, 0.587, 0.114])
        self.weight.weight.requires_grad = False

    def log_probs(self, images, shifted_images, k):
        difference = self.weight(shifted_images - images).squeeze(-1)
        probs = 0.5 * torch.sign(difference) * (2 * k - 1) + 0.5
        return probs.log()


class NeuralMaskGenerator(MaskGenerator):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.neural_operators = nn.ModuleList([UNet(in_channels=rgb_channels, out_channels=rgb_channels)
                                               for _ in range(n_classes)])
        self.register_parameter('log_sigma', nn.Parameter(torch.tensor(0.0)))

    def log_probs(self, images, shifted_images, k):
        images = images.permute(0, 3, 1, 2)
        shifted_images = shifted_images.permute(0, 3, 1, 2)
        difference = (shifted_images - self.neural_operators[k](images)) / self.log_sigma.exp()
        dependent_part = -0.5 * torch.einsum('bcij, bcij -> bij', difference, difference)
        constant_part = -rgb_channels * self.log_sigma
        return dependent_part + constant_part


# class GaussianMaskGenerator(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.n_classes = n_classes
#
#     def forward(self, batch):
#         images, shifted_images = batch
#         images = images.permute(0, 2, 3, 1)
#         shifted_images = shifted_images.permute(0, 2, 3, 1)
#         masks = []
#         data = torch.cat((images, shifted_images), dim=3).reshape(images.size(0), -1, 6).logit(eps=1e-3).cpu().numpy()
#         for color_pairs in data:
#             gm = GaussianMixture(self.n_classes)
#             gm.fit(color_pairs)
#             mask = gm.predict_proba(color_pairs).reshape(128, 128, self.n_classes)
#             masks.append(torch.Tensor(mask))
#         return torch.stack(masks, dim=0).permute(0, 3, 1, 2).to(images.device)


class GaussianMaskGenerator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, batch):
        images, shifted_images, predicted_masks = batch
        padded_images = torch.ones(4, images.size(0), 128, 128, device=images.device)
        padded_images[:3] = images.permute(1, 0, 2, 3).logit(eps=epsilon)
        padded_images = padded_images.permute(1, 0, 2, 3)
        shifted_images = shifted_images.logit(eps=epsilon)
        probs = predicted_masks.detach().clone().exp() + epsilon

        a_first_part = torch.einsum('uaij, ukij, ubij -> ukab', shifted_images, probs, padded_images)
        a_second_part = torch.einsum('uaij, ukij, ubij -> ukab', padded_images, probs, padded_images)
        normalization = torch.einsum('ukaa -> uk', a_second_part)
        a_first_part /= normalization.repeat(3, 4, 1, 1).permute(2, 3, 0, 1)
        a_second_part /= normalization.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        a_second_part += epsilon * torch.eye(4, device=images.device).repeat(images.size(0), self.n_classes, 1, 1)
        a = torch.einsum('ukab, ukbc -> ukac', a_first_part, a_second_part.inverse())
        # a = torch.solve(a_first_part.permute(0, 1, 3, 2), a_second_part).solution.permute(0, 1, 3, 2)

        colors_part = shifted_images.repeat(self.n_classes, 1, 1, 1, 1) - \
                      torch.einsum('ukab, ubij -> kuaij', a, padded_images)
        normalized_probs = probs / probs.sum(dim=(2, 3)).repeat(128, 128, 1, 1).permute(2, 3, 0, 1)
        sigma = torch.einsum('kuaij, ukij, kubij -> ukab', colors_part, normalized_probs, colors_part)
        sigma += epsilon * torch.eye(3, device=images.device).repeat(images.size(0), self.n_classes, 1, 1)

        log_masks = probs.log()
        log_masks += -0.5 * torch.einsum('kuaij, ukab, kubij -> ukij', colors_part, sigma.inverse(), colors_part)
        log_masks += -0.5 * torch.logdet(sigma).repeat(128, 128, 1, 1).permute(2, 3, 0, 1)
        return log_masks
