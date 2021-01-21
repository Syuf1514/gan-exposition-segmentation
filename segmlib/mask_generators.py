import torch

from torch import nn


rgb_channels = 3


def create_mask_generator(config):
    if config.mask_generator == 'scalar':
        return ScalarMaskGenerator(config.train_mask)
    elif config.mask_generator == 'affine':
        return AffineMaskGenerator(config.n_classes, config.sigma, config.train_mask)
    else:
        raise ValueError(f'unknown mask generator type "{config.mask_generator}"')


class ScalarMaskGenerator(nn.Module):
    def __init__(self, train):
        super().__init__()
        self.n_classes = 2
        self.operator = nn.Linear(rgb_channels, 1, bias=False)
        self.operator.weight.data = torch.tensor([[0.299, 0.587, 0.114]])
        for param in self.operator.parameters():
            param.requires_grad = train

    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.cat((self.operator(images), self.operator(shifted_images)), dim=3).permute(0, 3, 1, 2)
        return log_masks


class AffineMaskGenerator(nn.Module):
    def __init__(self, n_classes, sigma, train_sigma):
        super().__init__()
        self.n_classes = n_classes
        self.register_parameter('sigma', nn.Parameter(torch.tensor(sigma), requires_grad=train_sigma))
        self.operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels) for _ in range(n_classes)])

    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.criterion(operator(images), shifted_images)
                                 for operator in self.operators], dim=1)
        return log_masks

    def criterion(self, colors_hat, colors):
        return -torch.sum((colors_hat - colors)**2, dim=-1) / (2 * self.sigma**2)
