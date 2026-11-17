"""
Stochastic image transforms and small helpers used during stimulus synthesis.

Paper reference: Section 2.2.3 (Optimization Procedure).

Before every forward pass the stimulus is passed through a random rescaling
transform and two nested random spatial-jitter transforms. These
"transformation-robustness" tricks come from feature visualization
(Olah et al., 2017): they stop the optimizer from exploiting pixel-exact,
high-frequency patterns that only work at one position or scale, so the
synthesized motion is robust and visible to human observers.

Note on magnitudes: ``jitter(d)`` shifts the image by an integer drawn from
[-(d // 2), d - 1 - d // 2] pixels (e.g. d = 13 -> -6 ... +6 px).
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation

# lucent / lucentpatch imports are kept exactly as in the original code.
# lucentpatch ships inside the cloned `yhit` repository.
import lucentpatch
import lucentpatch.transform
from lucentpatch.render import render_vis
from lucentpatch.objectives import neuron, slow, tv_slow, intensity_preservation
from lucent.optvis.objectives import channel
from lucent.optvis import param
import lucent
import lucent.optvis
import lucent.optvis.transform


# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

ncolor, nt = 3, 11  # RGB channels, frames per DorsalNet input clip


def _roundup(value):
    return np.ceil(value).astype(int)


def random_scale(scales, decay=1.0):
    """Random zoom: resize the frame by a factor drawn from `scales`, then
    zero-pad back to the nominal size.

    With the paper's scales (0.25 * [0.9, 1.1]) the 164 x 164 stimulus is
    shrunk to roughly 41 x 41 pixels, centred in a 164 x 164 canvas. Random
    rescaling stops the optimizer from relying on one exact spatial scale.
    """
    def inner(image_t):
        scale = decay * np.random.choice(scales)
        shp = image_t.shape[2:]

        nominal_shape = [_roundup(max([max(scales), 1]) * d) for d in shp]
        scale_shape = [_roundup(scale * d / 2) * 2 for d in shp]

        mode = 'bilinear'
        if max(scales) < 1.0:
            # This is a downsampling op, so use area downsampling
            mode = 'area'

        scale_shape = [int(item) for item in scale_shape]

        im = F.interpolate(image_t, size=scale_shape, mode=mode)
        padding = [(nominal_shape[1] - im.shape[-1]) // 2, (nominal_shape[1] - im.shape[-1]) // 2,
                   (nominal_shape[0] - im.shape[-2]) // 2, (nominal_shape[0] - im.shape[-2]) // 2]
        im_padded = F.pad(im, padding)
        return im_padded

    return inner


def jitter(d, decay=1.0):
    """Random integer translation with wrap-around (pixels leaving one side
    re-enter on the other). Encourages translation robustness so features are
    not pinned to an exact pixel location."""
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = int(decay * np.random.choice(d))
        dy = int(decay * np.random.choice(d))
        return torch.roll(image_t, (dy - d // 2, dx - d // 2), (2, 3))

    return inner


def subpixel_jitter(decay=1.0):
    """Small continuous (sub-pixel) translation by bilinear mixing of
    neighbouring pixels. (Not used in the paper's configuration.)"""
    m = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.0)

    def inner(image_t):
        dx = decay * .01 * np.random.choice(100)
        dy = decay * .01 * np.random.choice(100)

        im = (dx * dy * image_t[:, :, :-1, :-1] +
              dx * (1 - dy) * image_t[:, :, 1:, :-1] +
              (1 - dx) * dy * image_t[:, :, :-1, 1:] +
              (1 - dx) * (1 - dy) * image_t[:, :, 1:, 1:])
        im = m(im)

        return im

    return inner


def crop(n):
    def inner(image_t):
        return image_t[:, :, n:-n, n:-n]

    return inner


def roll(n, decay=1.0):
    """Deterministic diagonal shift by `n` pixels (wrap-around).
    (Not used in the paper's configuration.)"""
    def inner(image_t):
        scaled_n = (int(n[0] * decay), int(n[1] * decay))
        return torch.roll(image_t, n, dims=[2, 3])

    return inner


def random_rotate(max_angle, decay=1.0):
    """Random rotation via lucent. (Not used in the paper's configuration.)"""
    scaled_angle = int(max_angle * decay)
    angles = list(range(-scaled_angle, scaled_angle + 1)) + 5 * [0]
    return lucent.optvis.transform.random_rotate(angles)


def transform(x, transforms):
    """Apply a list of 2D transforms to a video batch.

    The video tensor [B, C, T, H, W] is reshaped to [1, B*C*T, H, W] so the
    SAME random draw (scale / shift) is applied to every frame and every
    stimulus of the batch; the result is reshaped back to 5D.
    """
    x_shape = x.shape
    x = x.reshape(1, x_shape[0] * x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
    for t in transforms:
        x = t(x)
    return x.reshape(x_shape[0], x_shape[1], x_shape[2], x.shape[2], x.shape[3])


def clip(x):
    """Clip input pixel values to [-2, 2] before they enter DorsalNet."""
    return torch.clip(x, -2, 2)


def get_anim(result, size=4, norm=False):
    """Build a matplotlib animation from a [C, T, H, W] array (used to export GIFs)."""
    fig = plt.figure(figsize=(size, size))
    ax = plt.gca()
    im = plt.imshow([[1]])
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    rg = None
    if norm:
        rg = [result.min(), result.max()]

    nt_ = result.shape[1]

    def drawframe(n):
        if n >= nt_:
            n = nt_ - 1
        else:
            im.set_data(result.transpose((2, 3, 0, 1))[:, :, :, n])

        if norm:
            im.set_clim(rg)  # colour limits
        return (im)

    anim = animation.FuncAnimation(fig, drawframe, frames=nt, interval=100, blit=False)

    plt.close()
    return anim


# ----------------------------------------------------------------------------
# Decay schedules for the transform magnitudes over iterations
# ----------------------------------------------------------------------------
def decay_factor_linear(iteration, total_iterations, min_value=0.0):
    """Decay factor that decreases linearly from 1 to min_value."""
    return max(min_value, 1 - iteration / total_iterations)


def decay_factor_exp(iteration, total_iterations, min_value=0.0):
    return min_value + (1 - min_value) * np.exp(-5 * iteration / total_iterations)


def decay_factor_cos(iteration, total_iterations, min_value=0.0):
    return min_value + (1 - min_value) * (0.5 * (1 + np.cos(np.pi * iteration / total_iterations)))


def decay_factor_logistic(iteration,
                          total_iterations,
                          min_value=0.0,
                          steepness=20,
                          midpoint=0.25):
    """Logistic decay used in the paper (steepness = 20, midpoint = 0.25):
    ~1 at the start of optimization, dropping to ~0 after roughly the first
    quarter of the iterations. It scales the maximum jitter magnitude so that
    jitter is large early and vanishes later, leaving only rescaling."""
    fraction = iteration / total_iterations
    return (1.0 - min_value) - 1.0 / (1 + np.exp(-steepness * (fraction - midpoint)))
