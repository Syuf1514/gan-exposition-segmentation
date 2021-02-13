import argparse
import wandb
import pytorch_lightning as pl
import os
import warnings
import logging
import multiprocessing
import torch

from pathlib import Path

from segmlib import UnconditionalBigGAN, UNet, SegmentationModel, EMMaskGenerator, ColorsEMMaskGenerator


os.environ['WANDB_SILENT'] = 'true'
logging.getLogger('lightning').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
multiprocessing.set_start_method('spawn')
root_path = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation')
parser.add_argument('--wandb', default='online', help='wandb mode [online/offline/disabled]')
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

gan = UnconditionalBigGAN.load(run.config.gan_weights, run.config.gan_resolution, run.config.gan_device)
backbone_mask_generator = EMMaskGenerator(em_steps=1, alpha=None)
direction_mask_generator = ColorsEMMaskGenerator(em_steps=10, alpha=0.3)
backbone = UNet(in_channels=3, out_channels=run.config.n_classes)
if run.config.backbone_weights is not None:
    weights = torch.load(run.config.backbone_weights, map_location='cpu')
    backbone.load_state_dict(weights)
model = SegmentationModel(run, gan, backbone_mask_generator, direction_mask_generator, backbone)

trainer = pl.Trainer(
    logger=False,
    gpus=[run.config.model_device],
    max_epochs=run.config.epochs,
    default_root_dir=run.dir,
    weights_summary=None
)

if run.config.train_model:
    trainer.fit(model)
trainer.test(model, verbose=False)
