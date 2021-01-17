import numpy as np
import torch

from skimage.measure import label
from torch.nn.functional import conv1d

from .utils import rgb2gray


class MaskGenerator:
    def __init__(self, gan, direction, hparams):
        super().__init__()
        self.gan = gan
        self.direction = direction

        self.mask_size_bound = hparams.mask_size_bound
        self.maxes_filter = hparams.maxes_filter
        self.gan_batch_size = hparams.gan_batch_size
        self.z_shift = hparams.z_shift
        self.z_noise = hparams.z_noise
        self.mask_size_filter = hparams.mask_size_filter
        self.components_area_bound = hparams.components_area_bound
        self.connected_components = hparams.connected_components
        self.gan_device = hparams.gan_device

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
        masks = self._extract_masks(images, shifted_images)
        batch = [self._postprocessing(image, shifted_image, mask)
                 for image, shifted_image, mask in zip(images, shifted_images, masks)
                 if not self._reject_sample(image, shifted_image, mask)]
        return batch

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

    @staticmethod
    def _normalize_gan_output(tensor):
        return 0.5 * tensor + 0.5
