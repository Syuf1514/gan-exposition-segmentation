import torch


def accuracy(mask_hat, mask):
    result = torch.where(mask_hat == mask, 1.0, 0.0).mean()
    return result.item()


def binary_iou(mask_hat, mask):
    numerator = torch.min(mask_hat, mask).sum()
    denominator = torch.max(mask_hat, mask).sum()
    if denominator == 0:
        return 1.0
    result = numerator / denominator
    return result.item()


def binary_fbeta(mask_hat, mask, beta):
    true_positives = torch.sum((mask_hat == 1) & (mask == 1))
    false_positives = torch.sum((mask_hat == 1) & (mask == 0))
    false_negatives = torch.sum((mask_hat == 0) & (mask == 1))
    numerator = (1 + beta**2) * true_positives
    denominator = (1 + beta**2) * true_positives + beta**2 * false_negatives + false_positives
    if denominator == 0.0:
        if false_negatives == 0:
            return 1.0
        else:
            return 0.0
    result = numerator / denominator
    return result.item()
