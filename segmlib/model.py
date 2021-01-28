import torch
import pytorch_lightning as pl
import numpy as np
import itertools as tls
import pandas as pd

from pathlib import Path
from wandb import Image
from torch.utils.data import DataLoader

from .images_generator import ImagesGenerator
from .datasets import ImagesDataset, SegmentationDataset
from .metrics import accuracy, binary_iou, binary_fbeta


class SegmentationModel(pl.LightningModule):
    def __init__(self, run, gan, mask_generator, backbone):
        super().__init__()
        self.run = run
        self.mask_generator = mask_generator
        self.backbone = backbone
        self.hparams = dict(run.config)

        self.train_images_generator = ImagesGenerator(gan, self.hparams.train_embeddings, self.hparams)
        self.valid_images_generator = ImagesGenerator(gan, self.hparams.valid_embeddings, self.hparams)

        self.test_sets_names = [Path(path).resolve().stem for path in self.hparams.test_datasets]
        self.validation_batch = None
        self.labels_permutation = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_lr},
            {'params': self.mask_generator.parameters(), 'lr': self.hparams.mask_generator_lr}
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
            lambda epoch: 10.0 if epoch == 0 else self.hparams.lr_decay ** (epoch - 1),
            lambda epoch: 0.0 if epoch == 0 else self.hparams.lr_decay ** (epoch - 1)
        ])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, images):
        masks = self.backbone(images).argmax(dim=1)
        if self.labels_permutation is not None:
            masks = self._permute_labels(masks, self.labels_permutation)
        return masks

    def step(self, batch):
        images, shifted_images = batch
        generated_masks = self.mask_generator(batch)
        predicted_masks = self.backbone(images)
        reference_masks = generated_masks + predicted_masks
        loss = torch.logsumexp(predicted_masks, dim=1).mean() - \
               torch.logsumexp(reference_masks, dim=1).mean()
        return loss, (images, generated_masks, predicted_masks, reference_masks)

    def training_step(self, batch, batch_idx):
        loss, (images, generated_masks, predicted_masks, reference_masks) = self.step(batch)
        self.run.log({'Training Loss': loss}, commit=False)
        classes_priors = predicted_masks.detach().softmax(dim=1).mean(dim=(0, 2, 3)).cpu()
        self.run.log({f'class_{k} prior': prior.item() for k, prior in enumerate(classes_priors)})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, last_batch = self.step(batch)
        if self.validation_batch is None:
            self.validation_batch = last_batch
        self.run.log({'Validation Loss': loss})

    def on_validation_epoch_end(self):
        images, generated_masks, predicted_masks, reference_masks = \
            [tensor.cpu().numpy() for tensor in self.validation_batch]
        self.run.log({'Last Batch Segmentation': [
            Image(image.transpose(1, 2, 0), masks={
                'generated': {'mask_data': generated_mask.argmax(axis=0)},
                'predicted': {'mask_data': predicted_mask.argmax(axis=0)},
                'reference': {'mask_data': reference_mask.argmax(axis=0)}
            }) for image, generated_mask, predicted_mask, reference_mask in
            zip(images, generated_masks, predicted_masks, reference_masks)]
        })
        self.run.log({'Operators': self._cpu_params(self.mask_generator)})
        self.validation_batch = None

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        masks = masks.cpu()
        masks_hat = self(images).cpu()
        if self.labels_permutation is None:
            self.labels_permutation = self._optimize_permutation(masks_hat, masks)
            masks_hat = self._permute_labels(masks_hat, self.labels_permutation)
        metrics_batch = [{
                'acc': accuracy(mask_hat, mask),
                'iou': binary_iou(mask_hat, mask),
                'fb': binary_fbeta(mask_hat, mask, self.hparams.fbeta_beta)
        } for mask_hat, mask in zip(masks_hat, masks)]
        return metrics_batch

    def test_epoch_end(self, test_steps_outputs):
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
        train_set = ImagesDataset(self.train_images_generator, length=self.hparams.train_samples)
        train_loader = DataLoader(train_set, batch_size=self.hparams.model_batch_size, num_workers=1)
        return train_loader

    def val_dataloader(self):
        valid_set = ImagesDataset(self.valid_images_generator, length=self.hparams.valid_samples)
        valid_loader = DataLoader(valid_set, batch_size=self.hparams.model_batch_size, num_workers=1)
        return valid_loader

    def test_dataloader(self):
        test_sets = [SegmentationDataset(set_path, self.hparams.gan_resolution, mask_type=set_name)
                     for set_name, set_path in zip(self.test_sets_names, self.hparams.test_datasets)]
        test_loaders = [DataLoader(test_set, batch_size=self.hparams.model_batch_size, num_workers=8)
                        for test_set in test_sets]
        return test_loaders

    def _optimize_permutation(self, masks_hat, masks):
        permutations = list(tls.permutations(range(self.hparams.n_classes)))
        metrics = [accuracy(self._permute_labels(masks_hat, permutation), masks) for permutation in permutations]
        return permutations[np.argmax(metrics)]

    @staticmethod
    def _permute_labels(tensor, permutation):
        result = torch.zeros_like(tensor)
        for old_label, new_label in enumerate(permutation):
            result[tensor == old_label] = new_label
        return result

    @staticmethod
    def _cpu_params(module):
        state_dict_copy = {param_name: param_value.cpu().numpy()
                           for param_name, param_value in module.state_dict().items()}
        return state_dict_copy
