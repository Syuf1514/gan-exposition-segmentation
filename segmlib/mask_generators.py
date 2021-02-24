import torch


class EMMaskGenerator(torch.nn.Module):
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
        images, shifted_images, prior_masks = batch
        batch_size, rgb_channels, *image_shape = images.shape
        batch_size, n_classes, *image_shape = prior_masks.shape
        device = images.device

        generated_masks = torch.zeros_like(prior_masks)
        padded_images = torch.cat((images, torch.ones(batch_size, 1, *image_shape, device=device)), dim=1)
        for _ in range(self.em_steps):
            with torch.no_grad():
                generated_probs = torch.softmax(prior_masks + generated_masks, dim=1) + self.eps
                ops, sigmas = self.m_step(padded_images, shifted_images, generated_probs,
                                          n_classes, batch_size, rgb_channels, device)
            generated_masks = self.e_step(padded_images, shifted_images, ops, sigmas,
                                          n_classes, image_shape, batch_size)
        # generated_images = torch.einsum('ukab, ubij, ukij -> uaij', ops, padded_images, generated_masks.softmax(dim=1))
        return generated_masks
