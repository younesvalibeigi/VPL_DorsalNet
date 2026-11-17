"""
Initial stimulus pair, spatial mask and smoothing filter.

Paper reference: Section 2.2.3 (Optimization Procedure) and 2.2.4
(Post-Processing Pipeline).
"""

import numpy as np
import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')


def generate_x(var):
    """Create the starting stimulus pair and the fixed helper tensors.

    Returns
    -------
    x : torch.Tensor [nex, 3, 11, 164, 164]
        A single i.i.d. standard-normal spatiotemporal pattern, duplicated
        `nex` times (nex = 2 -> one pair). Both members start identical; one
        is labelled 0 and the other 1 by the classifier, and the optimization
        drives them apart. Three colour channels match DorsalNet's RGB input.
    mask : torch.Tensor [1, 1, 1, 164, 164]
        Circular soft-edged mask (sigmoidal fall-off, ~66-pixel radius) that
        restricts where the optimization can shape pixel structure.
    filt : torch.Tensor [3, 1, 1, 7, 7]
        Normalized Gaussian low-pass filter (half-width 3 px, sigma = 1.2 px),
        applied per colour channel.
    pad : int
        Padding for `filt` (= 3).
    target_contrast, target_mixed_contrast : float
        Target values for the contrast penalties (iii) and (iv).
    """
    seed = var.seed

    # nex: number of stimuli (2 = one pair); 3 colour channels; 11 frames;
    # 164 x 164 pixels
    nex, ncolor, nt, ny, nx = var.nex_in, 3, 11, 164, 164

    # One random stimulus, repeated across the nex dimension so that both
    # members of the pair start from the same pattern.
    x_single = torch.randn(1, ncolor, nt, ny, nx, device=device)
    x = x_single.repeat(nex, 1, 1, 1, 1)

    # Target contrasts for penalties (iii) and (iv), scaled to the frame size
    target_contrast = 1.3 * (128 / nx) ** 2
    target_mixed_contrast = 1.15 * (128 / nx) ** 2

    # Circular soft-edged mask (sigmoidal fall-off, ~66 px radius)
    xi, yi = np.meshgrid(np.arange(-nx // 2, nx // 2), np.arange(-nx // 2, nx // 2))
    mask = torch.tensor(1 - 1 / (1 + np.exp(- .2 * (xi ** 2 + yi ** 2 - 66 ** 2))), dtype=torch.float, device=device).reshape(1, 1, 1, nx, nx)

    # Gaussian smoothing filter (half-width 3 px, sigma 1.2 px, unit sum)
    oversaturation = 1.0
    width = 3
    sigma = 1.2
    xi_, yi_ = np.meshgrid(np.arange(-width, width + 1), np.arange(-width, width + 1))
    G = np.exp(-(xi_ ** 2 + yi_ ** 2) / 2 / sigma ** 2).astype(np.float32)
    G = G / G.sum()
    pad = (G.shape[0] - 1) // 2

    filt = torch.tensor(G, device=device).reshape(1, 1, 1, *xi_.shape)
    filt = torch.cat((filt, filt, filt), axis=0)  # one copy per colour channel

    return x, mask, filt, pad, target_contrast, target_mixed_contrast
