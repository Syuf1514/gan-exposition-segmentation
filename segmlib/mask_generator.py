import numpy as np
import torch

from skimage.measure import label
from torch.nn.functional import conv1d

from segmlib.utils import rgb2gray


class MaskGenerator:
    def __init__(self, gan, z_dim, config):
        super().__init__()
        self.gan = gan
        self.z_dim = z_dim

        self.direction = torch.load(config.direction)
        self.mask_size_bound = config.mask_size_bound
        self.maxes_filter = config.maxes_filter
        self.gan_batch_size = config.gan_batch_size
        self.z_shift = config.z_shift
        self.z_noise = config.z_noise
        self.gan_max_retries = config.gan_max_retries
        self.mask_size_filter = config.mask_size_filter
        self.components_area_bound = config.components_area_bound
        self.connected_components = config.connected_components
        self.fbeta_beta = config.fbeta_beta

        if config.embeddings is None:
            self.embeddings = torch.zeros(1, z_dim)
        else:
            self.embeddings = torch.load(config.embeddings)

    def _extract_masks(self, images, shifted_images):
        gray_images = rgb2gray(images)
        gray_shifted_images = rgb2gray(shifted_images)
        masks = torch.where(gray_shifted_images > gray_images, 1, 0)
        return masks

    def _reject_by_mask_size(self, mask):
        mask_size = mask.sum() / np.product(mask.shape)
        return mask_size > self.mask_size_bound

    def _reject_by_maxes(self, shifted_image):
        stats = torch.histc(shifted_image, bins=12, min=0, max=1)
        stats = conv1d(stats.view(1, 1, -1), torch.ones(1, 1, 3), padding=1)
        stats = stats.view(-1).numpy()
        maxes = np.r_[True, stats[1:] >= stats[:-1]] & np.r_[stats[:-1] >= stats[1:], True]
        maxes = np.nonzero(maxes)[0]
        return len(maxes) <= 1

    def _reject_sample(self, image, shifted_image, mask):
        return (self.mask_size_filter and self._reject_by_mask_size(mask)) or \
               (self.maxes_filter and self._reject_by_maxes(shifted_image))

    def _connected_components_postprocessing(self, mask):
        labels = label(mask)
        areas = np.bincount(labels.flatten()) / np.product(labels.shape)
        max_label_area = np.max(areas[1:])
        noise_labels = np.where(areas < self.components_area_bound * max_label_area)[0]
        processed_mask = np.where(np.isin(labels, noise_labels), 0, mask)
        return torch.from_numpy(processed_mask)

    def _postprocessing(self, image, shifted_image, mask):
        if self.connected_components:
            mask = self._connected_components_postprocessing(mask)
        return image, mask

    def __call__(self):
        z_idx = torch.randint(0, len(self.embeddings), (self.gan_batch_size,))
        z_codes = self.embeddings[z_idx] + self.z_noise * torch.randn(self.gan_batch_size, self.z_dim)
        shifted_z_codes = z_codes + self.z_shift * self.direction
        with torch.no_grad():
            images = self.gan(z_codes).cpu()
            images = self._normalize_gan_output(images)
            shifted_images = self.gan(shifted_z_codes).cpu()
            shifted_images = self._normalize_gan_output(shifted_images)
        masks = self._extract_masks(images, shifted_images)
        batch = [self._postprocessing(image, shifted_image, mask)
                 for image, shifted_image, mask in zip(images, shifted_images, masks)
                 if not self._reject_sample(image, shifted_image, mask)]
        return batch

    @staticmethod
    def _normalize_gan_output(tensor):
        return 0.5 * tensor + 0.5
