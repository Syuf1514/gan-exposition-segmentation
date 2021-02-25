import torch
import pytorch_lightning as pl
import numpy as np
import itertools as tls
import pandas as pd

from pathlib import Path
from wandb import Image
from torch.utils.data import DataLoader

from .datasets import GenerationDataset, SegmentationDataset
from .metrics import accuracy, binary_iou, binary_fbeta


class SegmentationModel(pl.LightningModule):
    def __init__(self, run, gan, mask_generator, backbone, shifted_backbone):
        super().__init__()
        self.run = run
        self.gan = gan
        self.mask_generator = mask_generator
        self.backbone = backbone
        self.shifted_backbone = shifted_backbone
        self.hparams = dict(run.config)

        if self.hparams.direction is None:
            direction = torch.randn(1, gan.dim_z)
            direction /= torch.norm(direction)
        else:
            direction = torch.load(self.hparams.direction, map_location='cpu')
        self.register_parameter('direction', torch.nn.Parameter(direction, requires_grad=self.hparams.train_direction))
        self.test_sets_names = [Path(path).resolve().stem for path in self.hparams.test_datasets]
        self.labeling = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: self.hparams.lr_decay ** epoch)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, images):
        masks = self.backbone(images).argmax(dim=1)
        if self.labeling is not None:
            masks = self._perform_labeling(masks, self.labeling)
        return masks

    def on_before_zero_grad(self, optimizer):
        self.direction.data /= torch.norm(self.direction.data)

    def step(self, batch):
        images, shifted_images = batch
        predicted_masks = self.backbone(images.sigmoid()).log_softmax(dim=1)
        shifted_masks = self.shifted_backbone(shifted_images.sigmoid()).log_softmax(dim=1)
        generated_masks = self.mask_generator((shifted_images, images, shifted_masks))
        penalty_masks = self.mask_generator((shifted_images, images, torch.zeros_like(predicted_masks)[:, :1]))
        reference_masks = shifted_masks + generated_masks
        normalized_reference = reference_masks.detach().log_softmax(dim=1)
        prediction_kl = torch.sum(normalized_reference.exp() * (normalized_reference - predicted_masks), dim=1).mean()
        likelihood = reference_masks.logsumexp(dim=1).mean()
        # likelihood = torch.sum(generated_masks.softmax(dim=1) * shifted_masks, dim=1).mean()
        sigmoid_images = images.sigmoid()
        normalized_images = (sigmoid_images - sigmoid_images.mean(dim=(0, 2, 3)).reshape(1, -1, 1, 1)) / \
                            sigmoid_images.std(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        sigmoid_shifted_images = shifted_images.sigmoid()
        normalized_shifted_images = (sigmoid_shifted_images - sigmoid_shifted_images.mean(dim=(0, 2, 3)).reshape(1, -1, 1, 1)) / \
                                    sigmoid_shifted_images.std(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        penalty = 1.0 * torch.mean((normalized_images - normalized_shifted_images) ** 2,
                                    dim=(1, 2, 3)).reciprocal().mean()
        # penalty = 1.5 * penalty_masks.mean()
        return (prediction_kl, likelihood, penalty), \
               (predicted_masks, shifted_masks, generated_masks, reference_masks)

    def training_step(self, batch, batch_idx):
        losses, masks = self.step(batch)
        prediction_kl, likelihood, penalty = losses
        predicted_masks, shifted_masks, generated_masks, reference_masks = masks
        self.run.log({
            'Prediction KL': prediction_kl.item(),
            'Likelihood': likelihood.item(),
            'Penalty': penalty.item()
        }, commit=False)
        classes_priors = predicted_masks.detach().exp().mean(dim=(0, 2, 3))
        self.run.log({f'class_{k}': prior.item() for k, prior in enumerate(classes_priors)})
        return penalty - likelihood + prediction_kl

    def validation_step(self, batch, batch_idx):
        losses, masks = self.step(batch)
        predicted_masks, shifted_masks, generated_masks, reference_masks = \
            [tensor.argmax(dim=1).cpu().numpy() for tensor in masks]
        images, shifted_images = [tensor.sigmoid().permute(0, 2, 3, 1).cpu().numpy() for tensor in batch]
        self.run.log({'Segmentation Examples': [
            Image(image, masks={
                'predicted': {'mask_data': predicted_mask},
                'shifted': {'mask_data': shifted_mask},
                'generated': {'mask_data': generated_mask},
                'reference': {'mask_data': reference_mask}
            }) for image, predicted_mask, shifted_mask, generated_mask, reference_mask in
            zip(images, predicted_masks, shifted_masks, generated_masks, reference_masks)]
        }, commit=False)
        self.run.log({'Shifts Examples': [
            Image(image, caption=caption)
            for original_image, shifted_image in zip(images, shifted_images)
            for caption, image in {'original': original_image,
                                   'shifted': shifted_image}.items()]
        })

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        masks = masks.cpu()
        masks_hat = self(images).cpu()
        if self.labeling is None:
            self.labeling = self._optimize_labeling(masks_hat, masks)
            masks_hat = self._perform_labeling(masks_hat, self.labeling)
        metrics_batch = [{
                'acc': accuracy(mask_hat, mask),
                'iou': binary_iou(mask_hat, mask),
                'fb': binary_fbeta(mask_hat, mask, self.hparams.fbeta_beta)
        } for mask_hat, mask in zip(masks_hat, masks)]
        return metrics_batch

    def test_epoch_end(self, test_steps_outputs):
        self.run.log({'labels': self.labeling})
        if len(self.test_sets_names) == 1:
            test_steps_outputs = [test_steps_outputs]
        for set_name, set_metrics_batches in zip(self.test_sets_names, test_steps_outputs):
            stats = pd.DataFrame(tls.chain(*set_metrics_batches)).describe().T[['count', 'mean', 'std']]
            stats['std'] /= np.sqrt(stats['count'])
            self.run.log({f'{set_name}_{metric_name}': mean_value
                          for metric_name, mean_value in stats['mean'].iteritems()})
            self.run.log({f'{set_name}_{metric_name}_std': std_value
                          for metric_name, std_value in stats['std'].iteritems()})

    def train_dataloader(self):
        dataset = GenerationDataset(self.gan, self.direction, self.hparams, length=self.hparams.train_samples)
        dataloader = DataLoader(dataset, batch_size=self.hparams.model_batch_size, num_workers=0)
        return dataloader

    def val_dataloader(self):
        dataset = GenerationDataset(self.gan, self.direction.data, self.hparams, length=self.hparams.model_batch_size)
        dataloader = DataLoader(dataset, batch_size=self.hparams.model_batch_size, num_workers=0)
        return dataloader

    def test_dataloader(self):
        datasets = [SegmentationDataset(set_path, self.hparams.gan_resolution, mask_type=set_name)
                    for set_name, set_path in zip(self.test_sets_names, self.hparams.test_datasets)]
        dataloaders = [DataLoader(dataset, batch_size=self.hparams.model_batch_size, num_workers=8)
                       for dataset in datasets]
        return dataloaders

    def _optimize_labeling(self, masks_hat, masks):
        labelings = list(tls.product([0, 1], repeat=self.hparams.n_classes))
        metrics = [sum(binary_iou(self._perform_labeling(mask_hat, labeling), mask)
                       for mask_hat, mask in zip(masks_hat, masks)) for labeling in labelings]
        return labelings[np.argmax(metrics)]

    @staticmethod
    def _perform_labeling(tensor, permutation):
        result = torch.zeros_like(tensor)
        for old_label, new_label in enumerate(permutation):
            result[tensor == old_label] = new_label
        return result
