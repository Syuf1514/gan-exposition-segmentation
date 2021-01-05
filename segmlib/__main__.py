
import argparse
import json
import torch

from pathlib import Path

from segmlib import weights_path
from segmlib.unet.unet_model import UNet
from segmlib.biggan.gan_load import make_big_gan
from .training import SegmentationTrainParams, train_segmentation, update_out_json, evaluate_all_wrappers


parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation training')
parser.add_argument('--args', help='path to the json file with all the arguments')

parser.add_argument('--out', default='results', help='output folder path')
parser.add_argument('--weights', default=str(weights_path / 'bigbigan.pth'), help='GAN weights path')
parser.add_argument('--direction', default=str(weights_path / 'direction.pth'), help='latent direction weights path')
parser.add_argument('--embeddings', help='dataset images embeddings path')
parser.add_argument('--z_noise', type=float, default=0.0, help='amplitude of z noise')

parser.add_argument('--images', nargs='*', default=[None], help='validation images directories')
parser.add_argument('--masks', nargs='*', default=[None], help='validation masks directories')

parser.add_argument('--device', type=int, default=0, help='device to train the segmentation model on')
parser.add_argument('--gen_devices', type=int, nargs='+', default=[0, 1], help='devices for GAN to generate samples on')
parser.add_argument('--seed', type=int, help='random seed')

for key, value in SegmentationTrainParams().__dict__.items():
    value_type = type(value) if key != 'synthezing' else str
    parser.add_argument(f'--{key}', type=value_type)

args = parser.parse_args()


if args.args is not None:
    json_args = json.load(open(args.args))
    namespace = argparse.Namespace()
    namespace.__dict__.update(json_args)
    args = parser.parse_args(namespace=namespace)

out = Path(args.out).resolve()
out.mkdir(exist_ok=True)

json.dump(args.__dict__, open(out / 'args.json', 'w'))


if args.seed is not None:
    torch.random.manual_seed(args.seed)
torch.cuda.set_device(args.device)

G = make_big_gan(args.weights).eval().cuda()
bg_direction = torch.load(args.direction)

model = UNet().train().cuda()
train_params = SegmentationTrainParams(**args.__dict__)

synthetic_score = train_segmentation(G, bg_direction, model, train_params, str(out), args.gen_devices,
                                     val_dirs=[args.images[0], args.masks[0]], zs=args.embeddings, z_noise=args.z_noise)

score_json = out / 'score.json'
update_out_json({'synthetic': synthetic_score}, score_json)
print(f'Synthetic data score: {synthetic_score}')

if args.val_images_dirs:
    evaluate_all_wrappers(model, score_json, args.images, args.masks)
