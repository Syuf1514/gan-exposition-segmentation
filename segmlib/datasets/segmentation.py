import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class SegmentationDataset(Dataset):
    def __init__(self, path, resolution, mask_type='default'):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise RuntimeError(f'dataset path "{self.path}" is not a valid folder')

        self.resolution = resolution
        if mask_type == 'flowers':
            self._mask_processing = self._flowers_mask_processing
        else:
            self._mask_processing = self._default_mask_processing

        images = {file.stem: file for file in self.path.glob('images/*')}
        masks = {file.stem: file for file in self.path.glob('masks/*')}
        if images.keys() != masks.keys():
            raise RuntimeError(f'images and masks filenames are not aligned in "{self.path}"')
        self.items = [(images[name], masks[name]) for name in images]

        self.basic_transforms = transforms.Compose([
            Image.open,
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        self.image_transforms = transforms.Compose([
            self.basic_transforms,
            self._image_processing
        ])
        self.mask_transforms = transforms.Compose([
            self.basic_transforms,
            self._mask_processing
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
    def _image_processing(tensor):
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) != 3:
            raise RuntimeError(f'expected image to have 1 or 3 channels, got {tensor.size(0)}')
        return tensor

    @staticmethod
    def _default_mask_processing(tensor):
        mask = torch.where(tensor[0] > 0.5, 1, 0)
        return mask

    @staticmethod
    def _flowers_mask_processing(tensor):
        mask = torch.where((tensor[0] < 0.1) & (tensor[1] < 0.1) & (tensor[2] > 0.9), 0, 1)
        return mask
