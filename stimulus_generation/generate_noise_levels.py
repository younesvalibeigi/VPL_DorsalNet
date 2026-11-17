"""
Graded series of stimulus pairs ("optimization levels").

Paper reference: Section 2.2.5 (Optimization-Level Generation).

To let observer performance span a broad range, a second pair of stimuli is
started from a fresh random pattern (two identical copies) and optimized so
that each member moves toward the corresponding member of the already
optimized target pair:

    member 1 -> activation of target stimulus 1
    member 2 -> activation of target stimulus 2

The distance is the Euclidean (L2) distance between the TARGET LAYER's full
activation maps (all channels, frames and spatial positions), together with
the same four auxiliary penalties as in Section 2.2.3. The evolving pair is
saved every `var.gen_nosie_interval` (50) iterations, giving a sequence that
runs from noisy, hard-to-discriminate pairs to target-like, easy ones.
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

from func_transforms import transform, clip
from post_stim_anal import post_optim_manipulations, show_animations_in_notebook, post_optim_normalization
from general_func import get_actv


def generate_noise_levels(model,
                          transforms,
                          outputs,
                          var,
                          actv_x,
                          x_n,
                          mask_n,
                          filt_n,
                          pad_n,
                          target_contrast_n,
                          target_mixed_contrast_n,
                          base_folder=None,
                          branch_folder=None,
                          this_folder=None):
    """Optimize a new noise pair `x_n` toward the fixed target activations `actv_x`.

    Parameters
    ----------
    transforms : list
        Transforms returned by optim_stim_multiObj (last iteration: jitter has
        decayed to zero, so only random rescaling remains). Used unchanged.
    actv_x : torch.Tensor
        Target-layer activations of the optimized target pair, computed once
        and held fixed.
    x_n, mask_n, filt_n, pad_n, target_contrast_n, target_mixed_contrast_n
        New random starting pair and helper tensors (from generate_x).

    Returns
    -------
    stimulus_list : list
        Spatially averaged target-layer activations at each saved level
        (used for the 3D PCA visualization).
    this_loss : list
        [distance, total loss] per iteration.
    loss_at_saved_images : list
        Total loss at each saved level.
    fullStimulus_list : list[torch.Tensor]
        The saved stimulus pairs themselves ([2, 3, 11, 164, 164] each).
    """
    actv = var.actv_layers[0]
    x_n.requires_grad_(True)
    optimizer = torch.optim.Adam([x_n], lr=var.gen_noise_lr)
    if var.gen_noise_sch_lr == True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=var.gen_step_size, gamma=var.gamma_noise)

    this_loss = []

    stimulus_list = []        # spatially averaged activations per saved level
    fullStimulus_list = []    # full stimulus pairs per saved level
    loss_at_saved_images = []

    # ------------------------------------------------------------------
    # Save the starting (pure noise) pair as level 1
    # ------------------------------------------------------------------
    start_cc = 0
    cc = 1
    if var.gen_noise_save_stim == True:
        if cc >= start_cc:
            if var.save_gif == True:
                images = post_optim_manipulations(x_n.clone().detach(), filt_n, pad_n)
                images_norm, mask_final = post_optim_normalization(images.cpu().detach().numpy())
                full_path = os.path.join(base_folder, branch_folder, this_folder)
                full_path = os.path.normpath(full_path)
                show_animations_in_notebook(images_norm * mask_final, full_path, generation_level=cc, grayScale=var.grayScale)
        activation_layer = get_actv(model, outputs, x_n.clone().detach(), actv, mask_n, transforms)
        actv_layer_avg = torch.mean(activation_layer, dim=(-2, -1))
        stimulus_list.append(actv_layer_avg.clone().detach())
        fullStimulus_list.append(x_n.clone().detach())
        cc = cc + 1
    else:
        stimulus_list.append(x_n.clone().detach())

    for i in tqdm(range(var.gen_noise_iter)):

        # Mask -> transforms (rescaling only) -> clip to [-2, 2] -> DorsalNet
        x_ = transform(x_n * mask_n, transforms)
        x_ = x_.to(device)
        _ = model(clip(x_)).to(device)

        actv_x_n = outputs[actv]

        n = outputs[actv].shape[0] // 2

        # Sanity checks on the activation map size
        if (actv == 'res1'):
            assert outputs[actv].shape[3] % 2 == 1
        if (actv == 's1'):
            assert outputs[actv].shape[3] % 2 == 0

        cz, cy, cx = outputs[actv].shape[2] // 2, outputs[actv].shape[3] // 2, outputs[actv].shape[4] // 2

        # --------------------------------------------------------------
        # L2 distance between each new stimulus and its target, computed on
        # the flattened full activation map of the target layer
        # --------------------------------------------------------------
        actv_x_flat = actv_x.view(actv_x.size(0), -1)
        actv_x_n_flat = actv_x_n.view(actv_x_n.size(0), -1)

        squared_dist = torch.sum((actv_x_flat - actv_x_n_flat) ** 2, dim=1)
        euclidean_dist = torch.sqrt(squared_dist)

        # --------------------------------------------------------------
        # Same four auxiliary penalties as in Section 2.2.3
        # --------------------------------------------------------------
        assert x_n.shape[1] == 3

        xp = F.conv3d(x_n, filt_n, groups=3, padding=(0, pad_n, pad_n)) * mask_n
        x_averaged = clip(xp).mean(axis=1).reshape(x_n.shape[0], x_n.shape[2], -1)
        x_staggered_flipped = torch.flip(torch.roll(xp, 1, 0), [2])
        x_mixed = clip(.707 * (xp + x_staggered_flipped)).mean(1).reshape(x_n.shape[0], x_n.shape[2], -1)

        loss = (euclidean_dist.mean() +
                var.mean_penalty * ((clip(x_n)) ** 2).mean() +
                var.color_penalty * (clip(xp).mean(axis=3).mean(axis=3) ** 2).mean() +
                var.contrast_penalty * (((x_averaged ** 2).mean(axis=2) - target_contrast_n) ** 2).mean() +
                var.mixed_contrast_penalty * (((x_mixed ** 2).mean(axis=2) - target_mixed_contrast_n) ** 2).mean())

        this_loss.append([float(euclidean_dist.mean()),
                          float(loss)])

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if var.gen_noise_sch_lr == True and i > var.start_scheduled_gen_noise_lr:
            scheduler.step()

        # --------------------------------------------------------------
        # Save the evolving pair every `gen_nosie_interval` iterations.
        # (i + 1) avoids re-saving iteration 0 (already saved above) and makes
        # the last save coincide with the final iteration.
        # --------------------------------------------------------------
        if ((var.gen_nosie_interval != None) and ((i + 1) % var.gen_nosie_interval == 0)):
            print('euclidean_dist', euclidean_dist)
            print('loss', loss)
            if var.gen_noise_save_stim == True:
                if cc >= start_cc:
                    if var.save_gif == True:
                        images = post_optim_manipulations(x_n.clone().detach(), filt_n, pad_n)
                        images_norm, mask_final = post_optim_normalization(images.cpu().detach().numpy())
                        full_path = os.path.join(base_folder, branch_folder, this_folder)
                        full_path = os.path.normpath(full_path)
                        show_animations_in_notebook(images_norm * mask_final, full_path, generation_level=cc, grayScale=var.grayScale)
                activation_layer = get_actv(model, outputs, x_n.clone().detach(), actv, mask_n, transforms)
                actv_layer_avg = torch.mean(activation_layer, dim=(-2, -1))
                stimulus_list.append(actv_layer_avg.clone().detach())
                fullStimulus_list.append(x_n.clone().detach())
                cc = cc + 1
                print('lr: ', optimizer.param_groups[0]['lr'])
            else:
                stimulus_list.append(x_n.clone().detach())
            loss_at_saved_images.append(loss)

    return stimulus_list, this_loss, loss_at_saved_images, fullStimulus_list
