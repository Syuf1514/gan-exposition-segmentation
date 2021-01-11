import torch
import itertools as tls
import numpy as np

from torch.utils.data import IterableDataset, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from .postprocessing import connected_components_filter
from .gan_mask_gen import MaskGenerator, it_mask_gen


class MaskGeneratorDataset(IterableDataset):
    def __init__(self, gan, direction, config, length):
        super().__init__()
        self.length = length
        mask_postprocessing = [connected_components_filter] if config.connected_components else []
        self.mask_generator = MaskGenerator(gan, direction, config, (), mask_postprocessing).cuda().eval()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError('single process data loading is recommended')
        iterator = tls.islice(tls.chain.from_iterable(zip(*self.mask_generator()) for _ in tls.count()), self.length)
        return iterator

    def __len__(self):
        return self.length


class SegmentationDataset(Dataset):
    def __init__(self, path, resolution, mask_type='default'):
        self.path = Path(path).resolve()
        self.resolution = resolution
        if mask_type == 'default':
            mask_processing = self._default_mask_processing
        elif mask_type == 'flowers':
            mask_processing = self._flowers_mask_processing
        else:
            raise ValueError(f'unknown mask type "{mask_type}"')

        images = {file.stem: file for file in self.path.glob('images/*')}
        masks = {file.stem: file for file in self.path.glob('masks/*')}
        if images.keys() != masks.keys():
            raise RuntimeError(f'images and masks filenames are not aligned in "{self.path}"')
        self.items = [(images[name], masks[name]) for name in images]

        self.image_transforms = transforms.Compose([
            Image.open,
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        self.mask_transforms = transforms.Compose([
            self.image_transforms,
            mask_processing
        ])

    def __getitem__(self, item):
        image, mask = self.items[item]
        return self.image_transforms(image), self.mask_transforms(mask)

    def __len__(self):
        return len(self.items)

    def as_image(self, item):
        image, mask = self[item]
        result = torch.cat((image, mask.unsqueeze(0).repeat(3, 1, 1)), dim=2)
        result = (255 * result).to(torch.uint8).permute(1, 2, 0)
        return Image.fromarray(result.numpy())

    @staticmethod
    def _default_mask_processing(tensor):
        if tensor.size(0) != 1:
            raise RuntimeError(f'expected to have one channel in mask, got {tensor.size(0)} '
                               '(set mask_type="flowers" if you are loading the flowers dataset)')
        mask = torch.where(tensor[0] > 0.5, 1, 0)
        return mask

    @staticmethod
    def _flowers_mask_processing(tensor):
        mask = torch.where((tensor[0] < 0.1) & (tensor[1] < 0.1) & (tensor[2] > 0.9), 0, 1)
        return mask
