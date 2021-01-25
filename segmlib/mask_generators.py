import torch

from torch import nn


rgb_channels = 3


def create_mask_generator(config):
    if config.mask_generator == 'scalar':
        return ScalarMaskGenerator(config.train_mask)
    elif config.mask_generator == 'affine':
        return AffineMaskGenerator(config.n_classes, config.sigma, config.train_mask)
    elif config.mask_generator == 'beta':
        return BetaMaskGenerator(config.n_classes, config.epsilon)
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
        log_sigma = torch.log(torch.tensor(sigma))
        self.register_parameter('log_sigma', nn.Parameter(log_sigma, requires_grad=train_sigma))
        self.operators = nn.ModuleList([nn.Linear(rgb_channels, rgb_channels) for _ in range(n_classes)])
        for operator in self.operators:
            operator.weight.data = torch.eye(rgb_channels)

    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.criterion(operator(images), shifted_images)
                                 for operator in self.operators], dim=1)
        return log_masks

    def criterion(self, colors_hat, colors):
        color_dists = torch.sum((colors_hat - colors)**2, dim=-1)
        return -color_dists / (2 * torch.exp(2 * self.log_sigma)) - rgb_channels * self.log_sigma


class BetaMaskGenerator(nn.Module):
    def __init__(self, n_classes, epsilon):
        super().__init__()
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.operators = nn.ModuleList([nn.Linear(rgb_channels, 2 * rgb_channels) for _ in range(n_classes)])

    def forward(self, batch):
        images, shifted_images = batch
        images = images.permute(0, 2, 3, 1)
        shifted_images = shifted_images.permute(0, 2, 3, 1)
        log_masks = torch.stack([self.criterion(operator(images), shifted_images)
                                 for operator in self.operators], dim=1)
        return log_masks

    def criterion(self, parameters, colors):
        parameters = torch.exp(parameters) + 1
        alphas, betas = parameters.split(rgb_channels, dim=3)
        colors = torch.clamp(colors, min=self.epsilon, max=1 - self.epsilon)
        dependent_part = (alphas - 1) * torch.log(colors) + (betas - 1) * torch.log(1 - colors)
        constant_part = torch.lgamma(alphas + betas) - torch.lgamma(alphas) - torch.lgamma(betas)
        return torch.sum(dependent_part + constant_part, dim=3)
