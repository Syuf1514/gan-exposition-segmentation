import argparse
import wandb
import pytorch_lightning as pl
import os
import warnings
import logging
import multiprocessing

from pathlib import Path

from segmlib import UnconditionalBigGAN, UNet, SegmentationModel


# os.environ['WANDB_SILENT'] = 'true'
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
multiprocessing.set_start_method('spawn')
root_path = Path(__file__).resolve().parents[1]


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

gan = UnconditionalBigGAN.load(run.config.weights, run.config.gan_resolution, run.config.gan_device).eval()
backbone = UNet(in_channels=3, out_channels=2)
model = SegmentationModel(run, gan, backbone)

trainer = pl.Trainer(
    logger=False,
    gpus=[run.config.model_device],
    max_epochs=run.config.epochs,
    default_root_dir=run.dir,
    weights_summary=None
)

trainer.fit(model)
trainer.test(model, verbose=False)
