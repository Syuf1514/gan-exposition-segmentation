import torch

from . import BigGAN


class UnconditionalBigGAN(torch.nn.Module):
    def __init__(self, gan):
        super().__init__()
        self.gan = gan
        self.dim_z = self.gan.dim_z

    def forward(self, z):
        classes = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        return self.gan(z, self.gan.shared(classes))

    @staticmethod
    def load(weights_path, resolution, device):
        config = make_biggan_config(resolution)
        weights = torch.load(weights_path, map_location='cpu')
        gan = BigGAN.Generator(**config)
        gan.load_state_dict(weights, strict=False)
        gan = UnconditionalBigGAN(gan).cuda(device)
        return gan


def make_biggan_config(resolution):
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}
    config = {
        'G_param': 'SN', 'D_param': 'SN',
        'G_ch': 96, 'D_ch': 96,
        'D_wide': True, 'G_shared': True,
        'shared_dim': 128, 'dim_z': dim_z_dict[resolution],
        'hier': True, 'cross_replica': False,
        'mybn': False, 'G_activation': torch.nn.ReLU(inplace=True),
        'G_attn': attn_dict[resolution],
        'norm_style': 'bn',
        'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
        'G_fp16': False, 'G_mixed_precision': False,
        'accumulate_stats': False, 'num_standing_accumulations': 16,
        'G_eval_mode': True,
        'BN_eps': 1e-04, 'SN_eps': 1e-04,
        'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution,
        'n_classes': 1000
    }
    return config
