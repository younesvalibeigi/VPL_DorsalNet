"""
3D (space x time) filters and the circular mask.

In the paper's pipeline only ``Filters3D.apply_mask`` is used: the same
soft circular mask (~66-px radius) as during stimulus synthesis is applied
to the stimuli before they are given to the ANN classifiers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal


class Filters3D:
    def __init__(self, device=None):
        """Initialize filter helper with a chosen device."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def gaussian_3d(self, x, sigma_space=1.0, sigma_time=0.5):
        """Separable 3D Gaussian blur (time, then y, then x)."""
        b, c, t, h, w = x.shape
        ks_space = int(2 * np.ceil(2 * sigma_space) + 1)
        ks_time = int(2 * np.ceil(2 * sigma_time) + 1)

        x_grid = torch.arange(ks_space, device=x.device) - ks_space // 2
        g_space = torch.exp(-0.5 * (x_grid ** 2) / sigma_space ** 2)
        g_space /= g_space.sum()

        t_grid = torch.arange(ks_time, device=x.device) - ks_time // 2
        g_time = torch.exp(-0.5 * (t_grid ** 2) / sigma_time ** 2)
        g_time /= g_time.sum()

        x = F.conv3d(
            x,
            g_time.view(1, 1, -1, 1, 1).expand(c, -1, -1, -1, -1),
            padding=(ks_time // 2, 0, 0),
            groups=c,
        )

        x = F.conv3d(
            x,
            g_space.view(1, 1, 1, -1, 1).expand(c, -1, -1, -1, -1),
            padding=(0, ks_space // 2, 0),
            groups=c,
        )

        x = F.conv3d(
            x,
            g_space.view(1, 1, 1, 1, -1).expand(c, -1, -1, -1, -1),
            padding=(0, 0, ks_space // 2),
            groups=c,
        )

        return x

    def temporal_gaussian(self, x, sigma_t=1.0):
        b, c, t, h, w = x.shape
        ks = int(2 * np.ceil(2 * sigma_t) + 1)
        t_grid = torch.arange(ks, device=x.device) - ks // 2
        g_time = torch.exp(-0.5 * (t_grid ** 2) / sigma_t ** 2)
        g_time /= g_time.sum()
        return F.conv3d(
            x,
            g_time.view(1, 1, -1, 1, 1).expand(c, -1, -1, -1, -1),
            padding=(ks // 2, 0, 0),
            groups=c,
        )

    def gabor_3d(self, x, freq=0.05, theta=0, sigma_space=8, sigma_time=2):
        b, c, t, h, w = x.shape

        y, x_grid, z = torch.meshgrid(
            torch.arange(-h // 2, h // 2, device=x.device),
            torch.arange(-w // 2, w // 2, device=x.device),
            torch.arange(-t // 2, t // 2, device=x.device),
            indexing='ij'
        )

        xr = x_grid * torch.cos(theta) + y * torch.sin(theta)

        g = torch.exp(
            -0.5 * ((xr ** 2 + y ** 2) / sigma_space ** 2 + z ** 2 / sigma_time ** 2)
        ) * torch.cos(2 * np.pi * freq * xr)

        g /= g.abs().sum()

        kernel = g.unsqueeze(0).unsqueeze(0)
        return F.conv3d(x, kernel.expand(c, -1, -1, -1, -1),
                        padding="same", groups=c)

    def butterworth_lowpass(self, x, cutoff=0.25, order=3):
        b, c, t, h, w = x.shape
        out = []
        for i in range(b):
            data = x[i].detach().cpu().numpy()
            b_butter, a_butter = signal.butter(order, cutoff, btype="low")
            filt = signal.filtfilt(b_butter, a_butter, data, axis=1)
            out.append(torch.tensor(filt, device=x.device))
        return torch.stack(out)

    def box_blur(self, x, kernel_size=3):
        c = x.shape[1]
        k = torch.ones((c, 1, kernel_size, kernel_size, kernel_size),
                       device=x.device) / (kernel_size ** 3)
        return F.conv3d(x, k, padding=kernel_size // 2, groups=c)

    def median_blur(self, x, kernel_size=3):
        pad = kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, kernel_size, 1)\
                       .unfold(3, kernel_size, 1)\
                       .unfold(4, kernel_size, 1)
        median, _ = patches.contiguous().view(*patches.shape[:5], -1).median(dim=-1)
        return median

    def pseudo_flow_blur(self, x, sigma=1.0):
        return self.gaussian_3d(x, sigma_space=0.8, sigma_time=1.5 * sigma)

    def apply_mask(self, x, radius=66):
        """Soft circular mask (sigmoidal fall-off) centred on the frame."""
        nx = x.shape[-1]
        xi, yi = np.meshgrid(np.arange(-nx // 2, nx // 2), np.arange(-nx // 2, nx // 2))
        mask = torch.tensor(
            1 - 1 / (1 + np.exp(-0.2 * (xi ** 2 + yi ** 2 - radius ** 2))),
            dtype=torch.float,
            device=self.device
        ).reshape(1, 1, 1, nx, nx)
        return mask * x
