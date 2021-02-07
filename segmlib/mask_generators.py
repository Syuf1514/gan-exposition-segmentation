import torch

from torch import nn


rgb_channels = 3


class AffineMaskGenerator(nn.Module):
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
        self.register_parameter('log_sigma', nn.Parameter(torch.tensor(2.0)))

    def log_probs(self, images, shifted_images, k):
        # difference = self.inv_sigma_operators[k](shifted_images - self.affine_operators[k](images))
        difference = (shifted_images - self.affine_operators[k](images)) / self.log_sigma.exp()
        dependent_part = -0.5 * torch.einsum('bijc, bijc -> bij', difference, difference)
        # constant_part = torch.slogdet(self.inv_sigma_operators[k].weight).logabsdet
        constant_part = -rgb_channels * self.log_sigma
        return dependent_part + constant_part

    def forward(self, batch):
        images, shifted_images, predicted_masks = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.log_probs(images, shifted_images, k) for k in range(self.n_classes)], dim=1)
        return log_masks


class EMMaskGenerator(nn.Module):
    def __init__(self, em_steps):
        super().__init__()
        self.em_steps = em_steps
        self.eps = torch.finfo(torch.float32).eps

    def m_step(self, padded_images, shifted_images, generated_probs, n_classes, batch_size, rgb_channels, device):
        ops_first_part = torch.einsum('uaij, ukij, ubij -> ukab', shifted_images, generated_probs, padded_images)
        ops_second_part = torch.einsum('uaij, ukij, ubij -> ukab', padded_images, generated_probs, padded_images)
        ops_norm_factor = torch.einsum('ukaa -> uk', ops_second_part).reshape(batch_size, n_classes, 1, 1)
        ops_first_part_normed = ops_first_part / ops_norm_factor.expand(*ops_first_part.shape)
        ops_second_part_normed = ops_second_part / ops_norm_factor.expand(*ops_second_part.shape) + self.eps * \
                                 torch.eye(rgb_channels + 1, device=device).repeat(batch_size, n_classes, 1, 1)
        ops = torch.einsum('ukab, ukbc -> ukac', ops_first_part_normed, ops_second_part_normed.inverse())

        probs_norm_factor = generated_probs.sum(dim=(2, 3)).reshape(batch_size, n_classes, 1, 1)
        predicted_probs_normed = generated_probs / probs_norm_factor.expand(*generated_probs.shape)
        difference = shifted_images.repeat(n_classes, 1, 1, 1, 1) - \
                     torch.einsum('ukab, ubij -> kuaij', ops, padded_images)
        sigmas = torch.einsum('kuaij, ukij, kubij -> ukab', difference, predicted_probs_normed, difference) + \
                 self.eps * torch.eye(rgb_channels, device=device).repeat(batch_size, n_classes, 1, 1)

        return ops, sigmas

    def e_step(self, padded_images, shifted_images, ops, sigmas, n_classes, image_shape, batch_size):
        difference = shifted_images.repeat(n_classes, 1, 1, 1, 1) - \
                     torch.einsum('ukab, ubij -> kuaij', ops, padded_images)
        log_masks = -0.5 * torch.einsum('kuaij, ukab, kubij -> ukij', difference, sigmas.inverse(), difference) + \
                    -0.5 * torch.logdet(sigmas).reshape(batch_size, n_classes, 1, 1).expand(-1, -1, *image_shape)
        return log_masks

    def forward(self, batch):
        images, shifted_images, predicted_masks = batch
        batch_size, rgb_channels, *image_shape = images.shape
        batch_size, n_classes, *image_shape = predicted_masks.shape
        device = images.device
        padded_images = torch.cat((images, torch.ones(batch_size, 1, *image_shape, device=device)), dim=1)

        generated_masks = torch.zeros_like(predicted_masks)
        for _ in range(self.em_steps):
            with torch.no_grad():
                generated_probs = torch.softmax(predicted_masks + generated_masks, dim=1) + self.eps
                ops, sigmas = self.m_step(padded_images, shifted_images, generated_probs,
                                          n_classes, batch_size, rgb_channels, device)
            generated_masks = self.e_step(padded_images, shifted_images, ops, sigmas,
                                          n_classes, image_shape, batch_size)
        return generated_masks
