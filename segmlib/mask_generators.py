import torch

from abc import ABC, abstractmethod
from torch import nn

from .unet import UNet


rgb_channels = 3


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
        else:
            raise ValueError(f'unknown mask generator "{name}"')


class AffineMaskGenerator(MaskGenerator):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.affine_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels) for _ in range(n_classes)])
        for operator in self.affine_operators:
            operator.weight.data = torch.eye(rgb_channels)
        # self.inv_sigma_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels, bias=False)
        #                                           for _ in range(n_classes)])
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
