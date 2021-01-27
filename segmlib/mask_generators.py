import torch
import numpy as np

from abc import ABC, abstractmethod
from torch import nn


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
        else:
            raise ValueError(f'unknown mask generator "{name}"')


class AffineMaskGenerator(MaskGenerator):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.affine_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels) for _ in range(n_classes)])
        self.inv_sigma_operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels, bias=False)
                                                  for _ in range(n_classes)])
        for operator in self.affine_operators:
            operator.weight.data = torch.eye(rgb_channels)
        for operator in self.inv_sigma_operators:
            operator.weight.data = np.sqrt(12.0) * torch.eye(rgb_channels)

    def log_probs(self, images, shifted_images, k):
        difference = self.inv_sigma_operators[k](shifted_images - self.affine_operators[k](images))
        dependent_part = -0.5 * torch.einsum('bijc, bijc -> bij', difference, difference)
        constant_part = torch.slogdet(self.inv_sigma_operators[k].weight).logabsdet
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
