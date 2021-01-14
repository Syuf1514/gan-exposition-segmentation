import argparse
import wandb
import pytorch_lightning as pl
import os
import warnings
import logging
import multiprocessing

from pathlib import Path
from torch.utils.data import DataLoader

from segmlib import root_path
from segmlib.unet import UNet
from segmlib.biggan import make_big_gan
from segmlib.dataset import MaskGeneratorDataset, SegmentationDataset
from segmlib.model import SegmentationModel
from segmlib.mask_generator import MaskGenerator


os.environ['WANDB_SILENT'] = 'true'
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
multiprocessing.set_start_method('spawn')


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

gan = make_big_gan(run.config.weights).eval().cuda(run.config.gan_device)
mask_generator = MaskGenerator(gan, gan.dim_z, run.config)

backbone = UNet(in_channels=3, out_channels=2)
test_sets_names = [Path(path).resolve().stem for path in run.config.test_datasets]

train_set = MaskGeneratorDataset(mask_generator, length=run.config.train_samples)
valid_set = MaskGeneratorDataset(mask_generator, length=run.config.valid_samples)
test_sets = [SegmentationDataset(path, resolution=128, mask_type=name)
             for name, path in zip(test_sets_names, run.config.test_datasets)]
train_loader = DataLoader(train_set, batch_size=run.config.model_batch_size, num_workers=1)
valid_loader = DataLoader(valid_set, batch_size=run.config.model_batch_size, num_workers=1)
test_loaders = [DataLoader(test_set, batch_size=run.config.model_batch_size, num_workers=8)
                for test_set in test_sets]

trainer = pl.Trainer(
    logger=False,
    gpus=[run.config.model_device],
    max_epochs=run.config.epochs,
    default_root_dir=run.dir,
    weights_summary=None
)

model = SegmentationModel(run, backbone, hparams=dict(run.config))
trainer.fit(model, train_loader, valid_loader)
metrics = trainer.test(model, test_loaders, verbose=False)
run.log({
    f'{set_name}_{metric_name}': metric_value
    for set_name, set_metrics in zip(test_sets_names, metrics)
    for metric_name, metric_value in set_metrics.items()
})
