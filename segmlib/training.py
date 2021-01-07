import os
import sys
import argparse
import json
import torch
import numpy as np
import wandb

import matplotlib
matplotlib.use("Agg")

from tqdm import tqdm
from pathlib import Path

from .unet.unet_model import UNet
from .biggan.gan_load import make_big_gan
from .utils.utils import to_image

from .gan_mask_gen import MaskGenerator, it_mask_gen
from .data import SegmentationDataset
from .metrics import model_metrics, IoU, accuracy, F_max
from .postprocessing import connected_components_filter, SegmentationInference, Threshold
from .visualization import overlayed
from segmlib import root_path


DEFAULT_EVAL_KEY = 'id'
THR_EVAL_KEY = 'thr'
SEGMENTATION_RES = 128


def train_segmentation(G, direction, model, run, val_dirs, mask_postprocessing):
    run_path = Path(wandb.run.dir).resolve()
    wandb.watch(model)
    model.train()

    batch_size = run.config.batch_size // len(run.config.gan_devices)
    mask_generator = MaskGenerator(G, direction, run.config, batch_size,
                                   mask_postprocessing=mask_postprocessing).cuda().eval()
    num_test_steps = run.config.test_samples_count // batch_size
    test_samples = [mask_generator() for _ in range(num_test_steps)]
    test_samples = [[s[0].cpu(), s[1].cpu()] for s in test_samples]

    optimizer = torch.optim.Adam(model.parameters(), lr=run.config.rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, run.config.steps_per_rate_decay, run.config.rate_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_step = 0
    checkpoint = run_path / 'checkpoint.pth'
    if checkpoint.is_file():
        start_step = load_checkpoint(model, optimizer, lr_scheduler, checkpoint)
        print('Starting from step {} checkpoint'.format(start_step))

    sample_generator = it_mask_gen(mask_generator, run.config.gan_devices, torch.cuda.current_device())
    iterator = tqdm(range(run.config.n_epochs), desc='Training')
    for epoch in iterator:
        img, ref = next(sample_generator)

        step = epoch + start_step
        model.zero_grad()
        prediction = model(img.cuda())
        loss = criterion(prediction, ref.cuda())

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step % run.config.steps_per_checkpoint_save == 0 and not run.disabled:
            save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint)

        if step % run.config.steps_per_log == 0:
            with torch.no_grad():
                loss = 0.0
                for img, ref in test_samples:
                    prediction = model(img.cuda())
                    loss += criterion(prediction, ref.cuda()).item()
            loss = loss / num_test_steps

            iterator.set_postfix_str(f'{type(criterion).__name__:.5}: {loss:.3}')

        if step % run.config.steps_per_validation == 0 and (val_dirs is not None):
            model.eval()
            eval_dict = evaluate(
                SegmentationInference(model, resize_to=SEGMENTATION_RES),
                val_dirs[0], val_dirs[1], (F_max,))
            update_out_json(eval_dict, run_path / 'score.json')
            model.train()
        if step % run.config.steps_per_log == 0:
            wandb.log({'Segmentation Examples': [
                wandb.Image(image.detach().cpu().numpy().transpose(1, 2, 0), masks={
                    'predictions': {
                        'mask_data': mask.detach().cpu().numpy(),
                        'class_labels': {
                            0: 'background',
                            1: 'object'
                        }
                    }
                }) for image, mask in zip(img[:run.config.n_examples], ref[:run.config.n_examples])
            ]})

    model.eval()
    if not run.disabled:
        torch.save(model.state_dict(), run_path / 'model.pth')


def evaluate_gan_mask_generator(model, G, direction, run, mask_postprocessing):
    batch_size = run.config.batch_size // len(run.config.gan_devices)
    mask_generator = MaskGenerator(G, direction, run.config, batch_size,
                                   mask_postprocessing=mask_postprocessing).cuda().eval()
    def it():
        while True:
            sample = mask_generator()
            for img, mask in zip(sample[0], sample[1]):
                yield img.unsqueeze(0), mask

    score = {
        **model_metrics(SegmentationInference(model), it(), run.config.test_samples_count, (F_max,)),
        **model_metrics(Threshold(model), it(), run.config.test_samples_count, (IoU, accuracy)),
    }
    return score


def save_checkpoint(model, opt, scheduler, step, checkpoint):
    state_dict = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step
    }
    torch.save(state_dict, checkpoint)


def load_checkpoint(model, opt, scheduler, checkpoint):
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    opt.load_state_dict(state_dict['opt'])
    scheduler.load_state_dict(state_dict['scheduler'])
    return state_dict['step']


def update_out_json(eval_dict, out_json):
    out_dict = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            out_dict = json.load(f)

    with open(out_json, 'w') as out:
        out_dict.update(eval_dict)
        json.dump(out_dict, out)


@torch.no_grad()
def evaluate(segmentation_model, images_dir, masks_dir, metrics, size=None):
    segmentation_dl = torch.utils.data.DataLoader(
        SegmentationDataset(images_dir, masks_dir, size=size, crop=False), 1, shuffle=False)

    eval_out = model_metrics(segmentation_model, segmentation_dl, stats=metrics)
    print('Segmenation model', eval_out)
    return eval_out


def keys_in_dict_tree(dict_tree, keys):
    for key in keys:
        if key not in dict_tree.keys():
            return False
        dict_tree = dict_tree[key]
    return True


@torch.no_grad()
def evaluate_all_wrappers(model, out_file, val_images_dirs, val_masks_dirs):
    model.eval()
    evaluation_dict = {}
    if os.path.isfile(out_file):
        with open(out_file, 'r') as f:
            evaluation_dict = json.load(f)

    for val_imgs, val_dirs in zip(val_images_dirs, val_masks_dirs):
        ds_name = val_imgs.split('/')[-2]
        print('\nEvaluating {}'.format(ds_name))
        if ds_name not in evaluation_dict.keys():
            evaluation_dict[ds_name] = {}

        if not keys_in_dict_tree(evaluation_dict, [ds_name, DEFAULT_EVAL_KEY]):
            evaluation_dict[ds_name][DEFAULT_EVAL_KEY] = evaluate(
                SegmentationInference(model, resize_to=SEGMENTATION_RES),
                val_imgs, val_dirs, (F_max,))
            update_out_json(evaluation_dict, out_file)

        if not keys_in_dict_tree(evaluation_dict, [ds_name, THR_EVAL_KEY]):
            evaluation_dict[ds_name][THR_EVAL_KEY] = evaluate(
                Threshold(model, resize_to=SEGMENTATION_RES), val_imgs, val_dirs, (IoU, accuracy))
            update_out_json(evaluation_dict, out_file)
