
import argparse
import torch
import wandb
import os

from pathlib import Path

from segmlib import root_path
from segmlib.unet.unet_model import UNet
from segmlib.biggan.gan_load import make_big_gan
from segmlib.training import train_segmentation, evaluate_all_wrappers, evaluate_gan_mask_generator
from segmlib.postprocessing import connected_components_filter


os.environ['WANDB_SILENT'] = 'true'


parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation')
parser.add_argument('--wandb', default='online', help='wandb mode, one of [online, offline, disabled]')
parser.add_argument('--config', default='configs/light.yaml', help='path to the yaml file with all the parameters')
run_args, other_args = parser.parse_known_args()

run = wandb.init(project='gan-exposition-segmentation', dir=root_path, mode=run_args.wandb, config=run_args.config)
parser = argparse.ArgumentParser()
for arg_name, arg_default in run.config.items():
    parser.add_argument(f'--{arg_name}', type=type(arg_default), default=arg_default)
params = parser.parse_args(other_args)
run.config.update(params, allow_val_change=True)

if run.config.seed is not None:
    torch.random.manual_seed(run.config.seed)
torch.cuda.set_device(run.config.device)

G = make_big_gan(run.config.weights).eval().cuda()
model = UNet().train().cuda()

val_dirs = [args.images[0], args.masks[0]] if run.config.images and run.config.masks else None
mask_postprocessing = [connected_components_filter] if run.config.connected_components else []
direction = torch.load(run.config.direction)

train_segmentation(G, direction, model, run, val_dirs, mask_postprocessing)
metrics = evaluate_gan_mask_generator(model, G, direction, run, mask_postprocessing)
wandb.log(metrics)

if val_dirs is not None:
    evaluate_all_wrappers(model, score_json, run.config.images, run.config.masks)
