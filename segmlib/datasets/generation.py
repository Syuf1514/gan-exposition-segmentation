import torch
import itertools as tls

from torch.utils.data import IterableDataset


class GenerationDataset(IterableDataset):
    def __init__(self, gan, direction, hparams, length):
        super().__init__()
        self.gan = gan
        self.direction = direction
        self.length = length

        if hparams.embeddings is None:
            self.embeddings = torch.zeros(1, gan.dim_z)
        else:
            self.embeddings = torch.load(hparams.embeddings)
        self.gan_device = hparams.gan_device
        self.gan_batch_size = hparams.gan_batch_size
        self.z_shift = hparams.z_shift
        self.z_noise = hparams.z_noise

    def generate(self):
        z_idx = torch.randint(0, len(self.embeddings), (self.gan_batch_size,))
        z_codes = self.embeddings[z_idx]
        z_codes += self.z_noise * torch.randn_like(z_codes)
        shifted_z_codes = z_codes + self.z_shift * self.direction.cpu()
        images = self.gan(z_codes.cuda(self.gan_device)).cpu()
        shifted_images = self.gan(shifted_z_codes.cuda(self.gan_device)).cpu()
        batch = [(image, shifted_image) for image, shifted_image in zip(images, shifted_images)]
        return batch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if (worker_info is not None) and (worker_info.num_workers > 1):
            raise RuntimeError('single process data loading is recommended')
        iterator = tls.islice(tls.chain.from_iterable(self.generate() for _ in tls.count()), self.length)
        return iterator

    def __len__(self):
        return self.length
