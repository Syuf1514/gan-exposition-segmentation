import argparse
import torch
import wandb
import os
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader

from segmlib import root_path
from segmlib.unet.unet_model import UNet
from segmlib.biggan.gan_load import make_big_gan
from segmlib.training import train_segmentation, evaluate_all_wrappers, evaluate_gan_mask_generator
from segmlib.postprocessing import connected_components_filter
from segmlib.dataset import MaskGeneratorDataset
from segmlib.model import SegmentationModel


os.environ['WANDB_SILENT'] = 'true'


parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation')
parser.add_argument('--wandb', default='online', help='wandb mode, one of [online, offline, disabled]')
parser.add_argument('--config', default='wandb/config.yaml', help='path to a file with default hyperparameters')
run_args, other_args = parser.parse_known_args()

run = wandb.init(project='gan-exposition-segmentation', dir=root_path, mode=run_args.wandb, config=run_args.config)
parser = argparse.ArgumentParser()
for arg_name, arg_default in run.config.items():
    parser.add_argument(f'--{arg_name}', type=type(arg_default), default=arg_default)
params = parser.parse_args(other_args)
run.config.update(params, allow_val_change=True)

if run.config.seed is not None:
    pl.seed_everything(run.config.seed)

gan = make_big_gan(run.config.weights).eval().cuda(0)
backbone = UNet().train().cuda(1)
direction = torch.load(run.config.direction)

train_set = MaskGeneratorDataset(gan, direction, run.config, length=run.config.train_samples_n)
valid_set = MaskGeneratorDataset(gan, direction, run.config, length=run.config.valid_samples_n)
train_loader = DataLoader(train_set, batch_size=run.config.model_batch_size)
valid_loader = DataLoader(valid_set, batch_size=run.config.model_batch_size)

trainer = pl.Trainer(
    logger=False,
    gpus=[1],
    max_epochs=run.config.epochs_n,
    default_root_dir=run.dir
)

model = SegmentationModel(run, backbone, hparams=dict(run.config))
trainer.fit(model, train_loader, valid_loader)

# metrics = evaluate_gan_mask_generator(backbone, gan, direction, run, mask_postprocessing)
# wandb.log(metrics)

# if val_dirs is not None:
#     evaluate_all_wrappers(backbone, run.config.images, run.config.masks)
