import torch
import pytorch_lightning as pl

from wandb import Image


class SegmentationModel(pl.LightningModule):
    class_labels = {
        0: 'background',
        1: 'object'
    }

    def __init__(self, run, backbone, hparams):
        super().__init__()
        self.run = run
        self.backbone = backbone
        self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        run.watch(backbone)

    def forward(self, x):
        segmentation = self.backbone(x)
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
        images, masks, masks_hat = [tensor.detach().cpu().numpy() for tensor in self.last_validation_batch]
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.steps_per_rate_decay, self.hparams.rate_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
