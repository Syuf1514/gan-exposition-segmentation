import numpy as np
import torch


class ImagesGenerator:
    def __init__(self, gan, direction, hparams):
        super().__init__()
        self.gan = gan
        self.direction = direction

        self.gan_device = hparams.gan_device
        self.gan_batch_size = hparams.gan_batch_size
        self.z_shift = hparams.z_shift
        self.z_noise = hparams.z_noise

        if hparams.embeddings is None:
            self.embeddings = torch.zeros(1, gan.dim_z)
        else:
            self.embeddings = torch.load(hparams.embeddings)

    def __call__(self):
        z_idx = torch.randint(0, len(self.embeddings), (self.gan_batch_size,))
        z_codes = self.embeddings[z_idx]
        z_codes += self.z_noise * torch.randn_like(z_codes)
        shifted_z_codes = z_codes + self.z_shift * self.direction
        with torch.no_grad():
            images = self.gan(z_codes.cuda(self.gan_device)).cpu()
            images = self._normalize_gan_output(images)
            shifted_images = self.gan(shifted_z_codes.cuda(self.gan_device)).cpu()
            shifted_images = self._normalize_gan_output(shifted_images)
        batch = [(image, shifted_image) for image, shifted_image in zip(images, shifted_images)]
        return batch

    @staticmethod
    def _normalize_gan_output(tensor):
        return 0.5 * tensor + 0.5
