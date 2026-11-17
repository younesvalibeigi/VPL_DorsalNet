"""
Class-based versions of the stimulus transforms and helpers.

In the paper's pipeline only ``StimulusTransforms.clip`` is used (in
prepare_model_in_out), to clip the masked stimuli to [-2, 2].
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation

# lucent / lucentpatch imports are kept as in the original code.
import lucent.optvis
import lucent.optvis.transform
from lucentpatch.render import render_vis
from lucentpatch.objectives import neuron, slow, tv_slow, intensity_preservation
from lucent.optvis.objectives import channel


class StimulusTransforms:
    """Transforms, jitters, crops, rotations, clipping, animation utilities
    and decay schedules for stimulus optimization."""

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def roundup(self, value):
        return np.ceil(value).astype(int)

    def random_scale(self, scales, decay=1.0):
        def inner(image_t):
            scale = decay * np.random.choice(scales)
            shp = image_t.shape[2:]

            nominal_shape = [self.roundup(max([max(scales), 1]) * d) for d in shp]
            scale_shape = [self.roundup(scale * d / 2) * 2 for d in shp]

            mode = "bilinear"
            if max(scales) < 1.0:
                mode = "area"

            scale_shape = [int(s) for s in scale_shape]
            im = F.interpolate(image_t, size=scale_shape, mode=mode)

            padding = [
                (nominal_shape[1] - im.shape[-1]) // 2,
                (nominal_shape[1] - im.shape[-1]) // 2,
                (nominal_shape[0] - im.shape[-2]) // 2,
                (nominal_shape[0] - im.shape[-2]) // 2,
            ]
            return F.pad(im, padding)

        return inner

    def jitter(self, d, decay=1.0):
        assert d > 1, "Jitter parameter must be >1"

        def inner(image_t):
            dx = int(decay * np.random.choice(d))
            dy = int(decay * np.random.choice(d))
            return torch.roll(image_t, (dy - d // 2, dx - d // 2), (2, 3))

        return inner

    def subpixel_jitter(self, decay=1.0):
        m = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.0)

        def inner(image_t):
            dx = decay * 0.01 * np.random.choice(100)
            dy = decay * 0.01 * np.random.choice(100)

            im = (
                dx * dy * image_t[:, :, :-1, :-1]
                + dx * (1 - dy) * image_t[:, :, 1:, :-1]
                + (1 - dx) * dy * image_t[:, :, :-1, 1:]
                + (1 - dx) * (1 - dy) * image_t[:, :, 1:, 1:]
            )
            return m(im)

        return inner

    def crop(self, n):
        def inner(image_t):
            return image_t[:, :, n:-n, n:-n]

        return inner

    def roll(self, n, decay=1.0):
        def inner(image_t):
            scaled = (int(n[0] * decay), int(n[1] * decay))
            return torch.roll(image_t, scaled, dims=[2, 3])

        return inner

    def random_rotate(self, max_angle, decay=1.0):
        scaled = int(max_angle * decay)
        angles = list(range(-scaled, scaled + 1)) + 5 * [0]
        return lucent.optvis.transform.random_rotate(angles)

    def apply_transforms(self, x, transforms):
        b, c, t, h, w = x.shape
        x = x.reshape(1, c * t, h, w)
        for tfn in transforms:
            x = tfn(x)
        return x.reshape(b, c, t, x.shape[-2], x.shape[-1])

    def clip(self, x):
        """Clip values to [-2, 2]."""
        return torch.clip(x, -2, 2)

    def get_anim(self, result, size=4, norm=False):
        fig = plt.figure(figsize=(size, size))
        ax = plt.gca()
        im = plt.imshow([[1]])
        plt.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(0, 0, 1, 1)

        rg = None
        if norm:
            rg = [result.min(), result.max()]

        nt = result.shape[1]

        def drawframe(n):
            if n >= nt:
                n = nt - 1
            im.set_data(result.transpose(2, 3, 0, 1)[:, :, :, n])
            if norm:
                im.set_clim(rg)
            return (im,)

        anim = animation.FuncAnimation(fig, drawframe, frames=nt, interval=100)
        plt.close()
        return anim

    def decay_linear(self, iteration, total, min_value=0.0):
        return max(min_value, 1 - iteration / total)

    def decay_exp(self, iteration, total, min_value=0.0):
        return min_value + (1 - min_value) * np.exp(-5 * iteration / total)

    def decay_cos(self, iteration, total, min_value=0.0):
        return min_value + (1 - min_value) * (
            0.5 * (1 + np.cos(np.pi * iteration / total))
        )

    def decay_logistic(self, iteration, total, min_value=0.0, steepness=20, midpoint=0.25):
        frac = iteration / total
        return (1.0 - min_value) - 1.0 / (1 + np.exp(-steepness * (frac - midpoint)))
