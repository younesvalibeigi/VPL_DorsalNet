"""
Post-processing that turns optimized stimuli into display-ready GIFs.

Paper reference: Section 2.2.4 (Post-Processing Pipeline).

    post_optim_manipulations  -> Gaussian smoothing (twice, with a centre crop
                                 to 128 x 128 in between) + soft circular mask
    post_optim_normalization  -> 2x spline up-sampling (128 -> 256), drop the
                                 first and last frame (11 -> 9), and normalize
                                 every frame to a fixed contrast and neutral
                                 gray luminance
    show_animations_in_notebook -> grayscale conversion, display remap and
                                 GIF export
"""

import os

import numpy as np
import scipy
import scipy.ndimage
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import HTML

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

from func_transforms import get_anim

ncolor, nt = 3, 11


def post_optim_manipulations(x, filt, pad):
    """Smooth, crop and mask an optimized stimulus batch.

    1. Depthwise convolution of every colour channel with the fixed Gaussian
       low-pass filter (removes pixel-level artifacts of stimulus generation).
    2. Centre-crop frames larger than 128 x 128 back to 128 x 128 so they
       match the mask below.
    3. Second pass of the same Gaussian filter (effective sigma ~1.7 px).
    4. Soft circular mask (~62-px radius on 128 x 128) so the stimulus has no
       sharp border that could act as an extra cue.
    """
    x_ = x.clone()
    x_ = F.conv3d(x_, filt, groups=3, padding=(0, pad, pad))  # each colour channel separately

    ny = x.shape[3]
    if ny > 128:
        dx = (ny - 128) // 2
        images = x_[:, :, :, dx:-dx, dx:-dx]
    else:
        images = x_

    images = F.conv3d(images, filt, groups=3, padding=(0, pad, pad))

    xi, yi = np.meshgrid(np.arange(-128 // 2, 128 // 2), np.arange(-128 // 2, 128 // 2))
    mask = torch.tensor(1 - 1 / (1 + np.exp(- .2 * (xi ** 2 + yi ** 2 - 62 ** 2))), device=device).reshape(1, 1, 1, 128, 128)

    images = images.to(device)
    images = images * mask

    return images


def show_animations_in_notebook(images, this_folder, group=None, generation_level=None, grayScale=False):
    """Save each stimulus of a batch as a GIF.

    With ``grayScale=True`` (paper) frames are converted with the standard
    luminance weights (0.2989 R + 0.5870 G + 0.1140 B) so colour cannot serve
    as a cue and observers must use motion. Pixel values are remapped for
    display as x -> 0.5 + 0.25 x and clipped to [0, 1].

    File name: ``level_<generation_level>_basic_animation_<i>.gif``.
    Level 0 is the fully optimized pair; levels 1, 2, ... are the graded
    series from generate_noise_levels.
    """
    html = ""
    nex = images.shape[0]
    for i in range(nex):
        if grayScale:
            gray_images = images[:, 0, :, :, :] * 0.2989 + images[:, 1, :, :, :] * 0.5870 + images[:, 2, :, :, :] * 0.1140
            gray_images = np.repeat(gray_images[:, np.newaxis, :, :, :], 3, axis=1)
            a = np.clip(.5 + .25 * gray_images[i, ...], 0, 1)
        else:
            a = np.clip(.5 + .25 * images[i, ...], 0, 1)
        anim = get_anim(a, 2)

        if generation_level != None:
            if group != None:
                anim.save(os.path.join(this_folder, 'G' + str(group) + '_L' + str(generation_level) + '_basic_animation_' + str(i) + '.gif'), fps=12)
                html += f"<video style='float:left' loop='loop' autoplay='autoplay' controls='controls'><source src='basic_animation_{i}.mp4'></video>"
            else:
                anim.save(os.path.join(this_folder, 'level_' + str(generation_level) + '_basic_animation_' + str(i) + '.gif'), fps=12)
                html += f"<video style='float:left' loop='loop' autoplay='autoplay' controls='controls'><source src='basic_animation_{i}.mp4'></video>"
        else:
            anim.save(os.path.join(this_folder, 'basic_animation_' + str(i) + '.gif'), fps=12)
            html += f"<video style='float:left' loop='loop' autoplay='autoplay' controls='controls'><source src='basic_animation_{i}.mp4'></video>"

    HTML(html)


def mean_stats_over_time(images):
    """Diagnostic plots of mean colour, intensity and contrast over frames."""
    plt.imshow(images.mean(axis=3).mean(axis=3).transpose((0, 2, 1)) + .5)
    plt.title('Mean color over time')
    plt.show()

    plt.plot((images.mean(axis=3).mean(axis=3).transpose((0, 2, 1)) + .5).mean(axis=2).T)
    plt.title('Mean intensity over time')
    plt.show()

    nt = images.shape[2]

    # Contrast of a composite of each stimulus with its (time-reversed) partner
    for i in range(images.shape[0]):
        composite = np.clip(.707 * images[i, :, :, ...] +
                            .707 * images[(i + 1) % images.shape[0], :, ::-1, ...], -2, 2)
        plt.plot(((composite.mean(axis=0) ** 2).reshape((nt, -1)).mean(axis=1)))
    plt.title('Composite contrast per frame (staggered)')
    plt.show()

    for i in range(images.shape[0]):
        composite = np.clip(images[i, ...], -2, 2)
        plt.plot(((composite.mean(axis=0) ** 2).reshape((nt, -1)).mean(axis=1)))
    plt.title('Contrast per frame (raw)')
    plt.show()


def normalize_frame(im, mask, desired_contrast):
    """Iteratively rescale one frame to `desired_contrast`, then shift it to
    neutral gray mean luminance. Two passes make sure both targets are met."""
    norm_im = im.copy()
    contrast = 0.0

    for i in range(2):
        # Gradient-descent-like scaling toward the desired contrast
        contrast = (np.clip(norm_im * mask, -2, 2).mean(axis=0) ** 2).mean()
        while abs(contrast - desired_contrast) > .0025:
            norm_im = norm_im * (1 + .5 * (desired_contrast / contrast - 1))
            contrast = (np.clip(norm_im * mask, -2, 2).mean(axis=0) ** 2).mean()

        # Shift toward neutral gray (0.5, 0.5, 0.5) mean colour
        mean_color = (np.clip(.5 + .25 * mask * norm_im, 0, 1)).sum(axis=1).sum(axis=1) / mask.size
        while max(abs(mean_color - np.array([.5, .5, .5]))) > .001:
            norm_im = norm_im - .5 * (mean_color - .5).reshape((3, 1, 1))
            mean_color = (np.clip(.5 + .25 * mask * norm_im, 0, 1)).sum(axis=1).sum(axis=1) / mask.size

    return norm_im


def post_optim_normalization(images):
    """Up-sample, trim boundary frames and normalize every frame.

    * 2x spline up-sampling (scipy.ndimage.zoom, 128 -> 256 px) enlarges the
      stimulus for presentation without pixelation.
    * Frames 0 and 10 are discarded (11 -> 9 frames): they receive incomplete
      temporal context in DorsalNet's 3D convolutions and are poorly
      constrained by the optimization.
    * Every frame is normalized to contrast 0.5 and neutral gray luminance,
      so res0 and res2 stimuli cannot differ in these low-level properties.

    Returns the normalized images and the 256 x 256 soft circular mask.
    """
    xi, yi = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
    vbig_mask = (1 - 1 / (1 + np.exp(- .2 * (xi ** 2 + yi ** 2 - (62 * 2) ** 2)))).reshape((1, 256, 256))

    images_hi = []
    for i in tqdm(range(images.shape[0])):
        images_hi.append(scipy.ndimage.zoom(images[i], [1, 1, 2, 2]))

    images_hi = np.stack(images_hi, axis=0)
    # Remove the first and last frame (11 -> 9 frames)
    images_hi = images_hi[:, :, 1:-1]

    images_norm = images_hi.copy()
    for i in range(images_hi.shape[0]):
        for j in range(images_hi.shape[2]):
            images_norm[i, :, j] = normalize_frame(images_norm[i, :, j], vbig_mask, 0.5)
    return images_norm, vbig_mask
