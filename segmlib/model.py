import torch
import pytorch_lightning as pl
import numpy as np
import itertools as tls
import pandas as pd

from pathlib import Path
from wandb import Image
from pytorch_lightning.metrics.functional import accuracy, iou, fbeta
from torch.utils.data import DataLoader

from segmlib.mask_generator import MaskGenerator
from segmlib.dataset import MaskGeneratorDataset, SegmentationDataset


class SegmentationModel(pl.LightningModule):
    class_labels = {
        0: 'background',
        1: 'object'
    }

    def __init__(self, run, gan, backbone, hparams):
        super().__init__()
        self.run = run
        self.backbone = backbone
        self.hparams = hparams

        self.train_mask_generator = MaskGenerator(gan, gan.dim_z, run.config)
        self.valid_mask_generator = MaskGenerator(gan, gan.dim_z, run.config)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_sets_names = [Path(path).resolve().stem for path in self.run.config.test_datasets]
        self.last_validation_batch = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.steps_per_rate_decay, self.hparams.rate_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def forward(self, images):
        segmentation = self.backbone(images).argmax(dim=1)
        return segmentation

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks_hat = self.backbone(images)
        loss = self.criterion(masks_hat, masks)
        self.run.log({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks_hat = self.backbone(images)
        loss = self.criterion(masks_hat, masks)
        self.run.log({'valid_loss': loss})
        self.last_validation_batch = images, masks, masks_hat

    def on_validation_epoch_end(self):
        images, masks, masks_hat = [tensor.cpu().numpy() for tensor in self.last_validation_batch]
        self.run.log({'Last Batch Segmentation': [
            Image(image.transpose(1, 2, 0), masks={
                'GAN': {
                    'mask_data': mask,
                    'class_labels': self.class_labels
                },
                'model': {
                    'mask_data': mask_hat.argmax(axis=0),
                    'class_labels': self.class_labels
                }
            }) for image, mask, mask_hat in zip(images, masks, masks_hat)]
        })
        self.last_validation_batch = None

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        masks = masks.cpu()
        masks_hat = self(images).cpu()
        metrics_batch = [{
                'acc': accuracy(mask_hat, mask).item(),
                'iou': iou(mask_hat, mask).item(),
                'fb': fbeta(mask_hat, mask, num_classes=len(self.class_labels),
                            beta=self.hparams.fbeta_beta, average='none')[1].item()
        } for mask_hat, mask in zip(masks_hat, masks)]
        return metrics_batch

    def test_epoch_end(self, outputs):
        if len(self.test_sets_names) == 1:
            outputs = [outputs]
        for set_name, set_metrics_batches in zip(self.test_sets_names, outputs):
            stats = pd.DataFrame(tls.chain(*set_metrics_batches)).describe().T[['count', 'mean', 'std']]
            stats['std'] /= np.sqrt(stats['count'])
            self.run.log({f'{set_name}_{metric_name}': mean_value
                          for metric_name, mean_value in stats['mean'].iteritems()})
            self.run.log({f'{set_name}_{metric_name}_std': std_value
                          for metric_name, std_value in stats['std'].iteritems()})

    def train_dataloader(self):
        train_set = MaskGeneratorDataset(self.train_mask_generator, length=self.run.config.train_samples)
        train_loader = DataLoader(train_set, batch_size=self.run.config.model_batch_size, num_workers=1)
        return train_loader

    def val_dataloader(self):
        valid_set = MaskGeneratorDataset(self.valid_mask_generator, length=self.run.config.valid_samples)
        valid_loader = DataLoader(valid_set, batch_size=self.run.config.model_batch_size, num_workers=1)
        return valid_loader

    def test_dataloader(self):
        test_sets = [SegmentationDataset(set_path, resolution=128, mask_type=set_name)
                     for set_name, set_path in zip(self.test_sets_names, self.run.config.test_datasets)]
        test_loaders = [DataLoader(test_set, batch_size=self.run.config.model_batch_size, num_workers=8)
                        for test_set in test_sets]
        return test_loaders
