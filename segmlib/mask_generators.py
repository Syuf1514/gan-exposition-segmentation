import torch


class EMMaskGenerator(torch.nn.Module):
    def __init__(self, em_steps, alpha, colors):
        super().__init__()
        self.em_steps = em_steps
        self.alpha = alpha
        self.colors = colors
        self.eps = torch.finfo(torch.float32).eps

    def gaussian_optimum(self, images, probs_normed, n_classes, image_shape, batch_size, rgb_channels, device):
        mus = torch.einsum('uaij, ukij -> kua', images, probs_normed)
        difference = images.repeat(n_classes, 1, 1, 1, 1) - \
                     mus.reshape(n_classes, batch_size, rgb_channels, 1, 1).expand(-1, -1, -1, *image_shape)
        sigmas = torch.einsum('kuaij, ukij, kubij -> ukab', difference, probs_normed, difference) + \
                 self.eps * torch.eye(rgb_channels, device=device).repeat(batch_size, n_classes, 1, 1)
        return mus, sigmas

    def gaussian_logprob(self, images, mus, sigmas, n_classes, image_shape, batch_size, rgb_channels):
        difference = images.repeat(n_classes, 1, 1, 1, 1) - \
                     mus.reshape(n_classes, batch_size, rgb_channels, 1, 1).expand(-1, -1, -1, *image_shape)
        log_masks = -0.5 * torch.einsum('kuaij, ukab, kubij -> ukij', difference, sigmas.inverse(), difference) + \
                    -0.5 * sigmas.logdet().reshape(batch_size, n_classes, 1, 1).expand(-1, -1, *image_shape)
        return log_masks

    def m_step(self, padded_images, shifted_images, generated_probs, n_classes, batch_size, image_shape, rgb_channels,
               device):
        ops_first_part = torch.einsum('uaij, ukij, ubij -> ukab', shifted_images, generated_probs, padded_images)
        ops_second_part = torch.einsum('uaij, ukij, ubij -> ukab', padded_images, generated_probs, padded_images)
        ops_norm_factor = torch.einsum('ukaa -> uk', ops_second_part).reshape(batch_size, n_classes, 1, 1)
        ops_first_part_normed = ops_first_part / ops_norm_factor.expand(*ops_first_part.shape)
        ops_second_part_normed = ops_second_part / ops_norm_factor.expand(*ops_second_part.shape) + self.eps * \
                                 torch.eye(rgb_channels + 1, device=device).repeat(batch_size, n_classes, 1, 1)
        shifted_ops = torch.einsum('ukab, ukbc -> ukac', ops_first_part_normed, ops_second_part_normed.inverse())

        probs_norm_factor = generated_probs.sum(dim=(2, 3)).reshape(batch_size, n_classes, 1, 1)
        predicted_probs_normed = generated_probs / probs_norm_factor.expand(*generated_probs.shape)
        if self.alpha is None:
            difference = shifted_images.repeat(n_classes, 1, 1, 1, 1) - \
                         torch.einsum('ukab, ubij -> kuaij', shifted_ops, padded_images)
            shifted_sigmas = torch.einsum('kuaij, ukij, kubij -> ukab', difference, predicted_probs_normed,
                                          difference) + \
                             self.eps * torch.eye(rgb_channels, device=device).repeat(batch_size, n_classes, 1, 1)
        else:
            shifted_sigmas = self.alpha**2 * torch.eye(rgb_channels, device=device).repeat(batch_size, n_classes, 1,
                                                                                             1)

        mus, sigmas = self.gaussian_optimum(padded_images[:, :rgb_channels, :, :], predicted_probs_normed,
                                            n_classes, image_shape, batch_size, rgb_channels, device)
        return (mus, sigmas), (shifted_ops, shifted_sigmas)

    def e_step(self, padded_images, shifted_images, params, n_classes, image_shape, batch_size, rgb_channels):
        (mus, sigmas), (shifted_ops, shifted_sigmas) = params
        difference = shifted_images.repeat(n_classes, 1, 1, 1, 1) - \
                     torch.einsum('ukab, ubij -> kuaij', shifted_ops, padded_images)
        shifted_masks = -0.5 * torch.einsum('kuaij, ukab, kubij -> ukij', difference, shifted_sigmas.inverse(),
                                            difference) + \
                        -0.5 * torch.logdet(shifted_sigmas).reshape(batch_size, n_classes, 1, 1).expand(-1, -1,
                                                                                                        *image_shape)
        images_masks = self.gaussian_logprob(padded_images[:, :rgb_channels, :, :], mus, sigmas, n_classes,
                                             image_shape, batch_size, rgb_channels)
        if self.colors:
            return images_masks + shifted_masks
        else:
            return shifted_masks

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
                params = self.m_step(padded_images, shifted_images, generated_probs,
                                     n_classes, batch_size, image_shape, rgb_channels, device)
            generated_masks = self.e_step(padded_images, shifted_images, params,
                                          n_classes, image_shape, batch_size, rgb_channels)

        (mus, sigmas), (shifted_ops, shifted_sigmas) = params
        generated_images = torch.einsum('ukab, ubij, ukij -> uaij', shifted_ops, padded_images, predicted_masks.exp())
        return generated_masks, generated_images
