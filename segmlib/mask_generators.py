import torch

from torch import nn


class AffineMaskGenerator(nn.Module):
    rgb_channels = 3

    def __init__(self, n_classes, sigma):
        super().__init__()
        self.n_classes = n_classes
        self.sigma = sigma
        self.operators = nn.ModuleList([nn.Linear(self.rgb_channels, self.rgb_channels)
                                        for _ in range(n_classes)])

    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.criterion(operator(images), shifted_images)
                                 for operator in self.operators], dim=1)
        return log_masks

    def criterion(self, colors_hat, colors):
        return -torch.sum((colors_hat - colors)**2, dim=-1) / (2 * self.sigma**2)


class BrightnessMaskGenerator(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.n_classes = 2
        self.sigma = sigma

    def forward(self, batch):
        images, shifted_images = batch
        log_masks = torch.stack((torch.zeros_like(gray_images), self.criterion(images, shifted_images)), dim=1)
        return log_masks

    def criterion(self, images, shifted_images):
        return self.rgb2gray(shifted_images - images) / self.sigma

    @staticmethod
    def rgb2gray(images):
        weights = torch.Tensor([0.299, 0.587, 0.114])
        gray = torch.einsum('bcij, c -> bij', images, weights)
        return gray
