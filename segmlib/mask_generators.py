import torch

from torch import nn


class EMMaskGenerator(nn.Module):
    def __init__(self, em_steps):
        super().__init__()
        self.em_steps = em_steps
        self.eps = torch.finfo(torch.float32).eps

    def m_step(self, padded_images, shifted_images, generated_probs,
               batch_size, n_directions, n_classes, rgb_channels, image_shape, device):
        ops_first_part = torch.einsum('usaij, ukij, ubij -> uskab', shifted_images, generated_probs, padded_images)
        ops_second_part = torch.einsum('uaij, ukij, ubij -> ukab', padded_images, generated_probs, padded_images)
        ops_norm_factor = torch.einsum('ukaa -> uk', ops_second_part).reciprocal()
        ops_first_part_normed = torch.einsum('uskab, uk -> uskab', ops_first_part, ops_norm_factor)
        ops_second_part_normed = torch.einsum('ukab, uk -> ukab', ops_second_part, ops_norm_factor) + \
            self.eps * torch.eye(rgb_channels + 1, device=device).repeat(batch_size, n_classes, 1, 1)
        ops = torch.einsum('uskab, ukbc -> uskac', ops_first_part_normed, ops_second_part_normed.inverse())

        probs_norm_factor = generated_probs.sum(dim=(2, 3)).reciprocal()
        predicted_probs_normed = torch.einsum('ukij, uk -> ukij', generated_probs, probs_norm_factor)
        difference = shifted_images.unsqueeze(2) - torch.einsum('uskab, ubij -> uskaij', ops, padded_images)
        sigmas = torch.einsum('uskaij, ukij, uskbij -> uskab', difference, predicted_probs_normed, difference) + \
            self.eps * torch.eye(rgb_channels, device=device).repeat(batch_size, n_directions, n_classes, 1, 1)

        return ops, sigmas

    def e_step(self, padded_images, shifted_images, ops, sigmas,
               batch_size, n_directions, n_classes, rgb_channels, image_shape, device):
        difference = shifted_images.unsqueeze(2) - torch.einsum('uskab, ubij -> uskaij', ops, padded_images)
        log_masks = -0.5 * torch.einsum('uskaij, uskab, uskbij -> ukij', difference, sigmas.inverse(), difference) + \
                    -0.5 * sigmas.logdet().sum(dim=1).reshape(batch_size, n_classes, 1, 1)
        return log_masks

    def forward(self, batch):
        images, shifted_images, predicted_masks = batch
        batch_size, n_directions, rgb_channels, *image_shape = shifted_images.shape
        batch_size, n_classes, *image_shape = predicted_masks.shape
        device = images.device

        generated_masks = torch.zeros_like(predicted_masks)
        padded_images = torch.cat((images, torch.ones(batch_size, 1, *image_shape, device=device)), dim=1)
        for _ in range(self.em_steps):
            with torch.no_grad():
                generated_probs = torch.softmax(predicted_masks + generated_masks, dim=1) + self.eps
                ops, sigmas = self.m_step(padded_images, shifted_images, generated_probs,
                                          batch_size, n_directions, n_classes, rgb_channels, image_shape, device)
            generated_masks = self.e_step(padded_images, shifted_images, ops, sigmas,
                                          batch_size, n_directions, n_classes, rgb_channels, image_shape, device)
        return generated_masks
