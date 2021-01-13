import torch


def rgb2gray(images):
    weights = torch.Tensor([0.299, 0.587, 0.114])
    gray = torch.einsum('bcij, c -> bij', images, weights)
    return gray
