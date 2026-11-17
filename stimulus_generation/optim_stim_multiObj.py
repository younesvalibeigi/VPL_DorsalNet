"""
Synthesis of a stimulus pair that targets a single DorsalNet layer.

Paper reference: Sections 2.2.2 (Linear Readout Classifiers) and 2.2.3
(Optimization Procedure).

DorsalNet is frozen; only the stimulus pixels are optimized. Each iteration:

1. Mask the pair, apply the stochastic transforms (two nested random jitters
   whose size decays logistically, plus random rescaling), clip pixel values
   to [-2, 2] and run DorsalNet.
2. For each residual layer, read the central spatial unit (32 x 11 = 352
   features) and pass it through that layer's fixed PCA classifier.
3. Combine the classifier terms (weight +1 for the target layer, 0 for the
   others) with the four auxiliary penalties and take an Adam step on x.

The classifier term is ``CEoutput = -BCE(sigmoid(w . a), labels)``.
Minimizing it maximizes the BCE, which pushes the two stimuli of the pair to
opposite extremes along the fixed classifier direction, i.e. makes them
maximally separable for the target layer.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# lucent / lucentpatch imports are kept as in the original code.
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

from classifier_objects import classifier_objects
from func_transforms import random_scale, jitter, transform, clip, decay_factor_logistic
from post_stim_anal import post_optim_manipulations, show_animations_in_notebook, post_optim_normalization
from general_func import get_actv


class Classifier:
    """Container for one layer's fixed readout classifier."""
    linclassifier: any       # torch.nn.Linear (frozen)
    CEloss: any              # BCE loss
    linlabels: any           # labels (0 for stimulus 1, 1 for stimulus 2)
    classifier_weights: any  # list with the weight tensor
    classifier_bias: any     # bias tensor
    control_actv: any        # +1 target layer, -1 inhibited layer, 0 uncontrolled


def optim_stim_multiObj(model,
                        outputs,
                        var,
                        x,
                        mask,
                        filt,
                        pad,
                        target_contrast,
                        target_mixed_contrast,
                        CEcoeff=1,
                        arr_PCAs=None,
                        crossEntFeat=2,
                        base_folder=None,
                        branch_folder=None,
                        this_folder=None):
    """Optimize stimulus pair `x` so the target layer (var.actv_layers[0])
    separates its two members along its fixed PCA direction.

    Returns
    -------
    stimulus_list : list[torch.Tensor]
        The optimized pair [2, 3, 11, 164, 164] (one entry).
    this_loss : list
        Per-iteration loss terms: 4 classifier terms (res0-res3), 4 penalties,
        total loss.
    loss_at_saved_images : list
        Total loss at the saved stimulus.
    classifiers : list[Classifier]
        The four fixed classifiers (used for the 3D PCA plots).
    transforms : list
        Transforms of the LAST iteration (jitter has decayed to zero, so only
        random rescaling remains). They are re-used by
        generate_noise_levels and get_actv.
    """
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=var.frw_lr)
    if var.scheduled_frw_lr == True:
        # Step-decay schedule: x0.5 every 100 scheduler steps, stepping only
        # after iteration `start_scheduled_frw_lr` (1200)
        scheduler_frw = torch.optim.lr_scheduler.StepLR(optimizer, step_size=var.step_size_frw, gamma=var.frw_gamma)
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        print(f"Initial learning rate: {current_lr}")

    layers = ['res0', 'res1', 'res2', 'res3']

    # ------------------------------------------------------------------
    # One fixed linear classifier per residual layer (Section 2.2.2)
    # ------------------------------------------------------------------
    classifiers = [Classifier() for _ in range(len(layers))]

    for index in range(len(layers)):
        pca_components = arr_PCAs[layers[index]]
        linclassifier, CEloss, linlabels, classifier_bias = classifier_objects(x.shape[0],
                                                                               var.weights,
                                                                               pca_num=var.pca_num,
                                                                               act=layers[index],
                                                                               pca_components=pca_components,
                                                                               random_seed=var.seed,
                                                                               requireGrad=var.requireGrad,
                                                                               loss_func=var.loss_func,
                                                                               crossEntFeat=crossEntFeat,
                                                                               central_pixel=var.central_pixel)
        classifiers[index].linclassifier = linclassifier.to(device)
        classifiers[index].CEloss = CEloss.to(device)
        classifiers[index].linlabels = linlabels.to(device)
        classifiers[index].classifier_weights = []
        classifiers[index].classifier_bias = classifier_bias
        # control_actv sets the sign of each layer's classifier term:
        # +1 target layer (separate the pair), -1 inhibited layer, 0 ignored
        if layers[index] in var.actv_layers:
            classifiers[index].control_actv = +1
        elif layers[index] in var.inhb_layers:
            classifiers[index].control_actv = -1
        else:
            classifiers[index].control_actv = 0
        classifiers[index].classifier_weights.append(linclassifier.weight)

    this_loss = []
    stimulus_list = []

    # ==================================================================
    # Target-pair optimization (Section 2.2.3)
    # ==================================================================
    for i in tqdm(range(var.frw_iterations)):

        # --------------------------------------------------------------
        # Stochastic transforms. The maximum jitter is scaled by a logistic
        # decay (steepness 20, midpoint 0.25): large early, zero later, so
        # that only random rescaling remains for the rest of the run.
        # --------------------------------------------------------------
        decay = decay_factor_logistic(i,
                                      var.frw_iterations,
                                      min_value=0.0,
                                      steepness=var.frw_logistic_func_steepness,
                                      midpoint=var.frw_logistic_func_midpoint)
        param_jitter_2 = int(np.round(13 * decay))
        param_jitter_1 = int(param_jitter_2 / 2)

        if param_jitter_2 < 2:
            transforms = [random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0)]
        elif param_jitter_1 < 2:
            transforms = [random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0),
                          jitter(param_jitter_2, decay=1.0)]
        else:
            transforms = [jitter(param_jitter_1, decay=1.0),
                          random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0),
                          jitter(param_jitter_2, decay=1.0)]

        # Mask -> transforms -> clip pixel values to [-2, 2] -> frozen DorsalNet
        x_ = transform(x * mask, transforms)
        x_ = x_.to(device)
        _ = model(clip(x_)).to(device)

        # --------------------------------------------------------------
        # Classifier term for every layer (central spatial unit)
        # --------------------------------------------------------------
        CEoutput_layers = []
        cosine_similarity_layers = []
        for index in range(len(layers)):
            cz, cy, cx = outputs[layers[index]].shape[2] // 2, outputs[layers[index]].shape[3] // 2, outputs[layers[index]].shape[4] // 2

            if var.central_pixel:
                # [nex, 32 channels, 11 frames] at the central unit -> [nex, 352]
                LCinput = classifiers[index].linclassifier(outputs[layers[index]][:, :, :, cy, cx].reshape((var.nex_in, -1)))
            else:
                LCinput = classifiers[index].linclassifier(outputs[layers[index]].reshape(var.nex_in, -1))

            if var.loss_func == 'CrossEntropyLoss':
                CEoutput = - classifiers[index].CEloss(LCinput, classifiers[index].linlabels)

            elif var.loss_func == 'BCELoss':
                sig_func = nn.Sigmoid()

                LCinput = LCinput.squeeze()
                classifiers[index].linlabels = classifiers[index].linlabels.float()
                # Negative BCE: minimizing it drives the pair apart along the
                # fixed classifier direction
                CEoutput = - classifiers[index].CEloss(sig_func(LCinput), classifiers[index].linlabels)

                # Pairwise activation distance (computed for monitoring only;
                # not part of the objective)
                A = outputs[layers[index]].reshape(outputs[layers[index]].shape[0], -1)  # [2, D]
                sim_loss = (A[0] - A[1]).pow(2).mean()
                cosine_similarity_layers.append(sim_loss)

            CEoutput_layers.append(CEoutput)

        # --------------------------------------------------------------
        # Auxiliary penalties (Section 2.2.3)
        # --------------------------------------------------------------
        assert x.shape[1] == 3

        # Smoothed (Gaussian-filtered) and masked stimulus
        xp = F.conv3d(x, filt, groups=3, padding=(0, pad, pad)) * mask

        # Average over colour channels -> [nex, 11, H*W]
        x_averaged = clip(xp).mean(axis=1).reshape(x.shape[0], x.shape[2], -1)

        # Composite of the two pair members: the partner is taken by rolling
        # the batch by one and flipping it in time, then superimposed.
        x_staggered_flipped = torch.flip(torch.roll(xp, 1, 0), [2])
        x_mixed = clip(.707 * (xp + x_staggered_flipped)).mean(1).reshape(x.shape[0], x.shape[2], -1)

        # Total objective. control_actv: +1 target layer, 0 other layers.
        # Penalties: (i) mean squared pixel -> gray, (ii) colour cast,
        # (iii) per-frame contrast -> target, (iv) composite contrast -> target.
        loss = (CEoutput_layers[0] * CEcoeff * classifiers[0].control_actv +
                CEoutput_layers[1] * CEcoeff * classifiers[1].control_actv +
                CEoutput_layers[2] * CEcoeff * classifiers[2].control_actv +
                CEoutput_layers[3] * CEcoeff * classifiers[3].control_actv +
                var.mean_penalty * ((clip(x)) ** 2).mean() +
                var.color_penalty * (clip(xp).mean(axis=3).mean(axis=3) ** 2).mean() +
                var.contrast_penalty * (((x_averaged ** 2).mean(axis=2) - target_contrast) ** 2).mean() +
                var.mixed_contrast_penalty * (((x_mixed ** 2).mean(axis=2) - target_mixed_contrast) ** 2).mean())

        this_loss.append([float(CEoutput_layers[0]), float(CEoutput_layers[1]), float(CEoutput_layers[2]), float(CEoutput_layers[3]),
                          float(var.mean_penalty * ((clip(x)) ** 2).mean()),
                          float(var.color_penalty * (clip(xp).mean(axis=3).mean(axis=3) ** 2).mean()),
                          float(var.contrast_penalty * (((x_averaged ** 2).mean(axis=2) - target_contrast) ** 2).mean()),
                          float(var.mixed_contrast_penalty * (((x_mixed ** 2).mean(axis=2) - target_mixed_contrast) ** 2).mean()),
                          float(loss)])

        # Adam step on the stimulus pixels only
        optimizer.zero_grad()
        loss.backward()
        if var.clip_frw is True:
            torch.nn.utils.clip_grad_norm_([x], max_norm=var.max_norm_frw)

        optimizer.step()

        if var.scheduled_frw_lr == True and i > var.start_scheduled_frw_lr:
            scheduler_frw.step()

    # ------------------------------------------------------------------
    # Store the fully optimized pair
    # ------------------------------------------------------------------
    loss_at_saved_images = []

    if var.save_stim_atIter == True:
        cc = 0
        if var.save_gif == True:
            images = post_optim_manipulations(x.clone().detach(), filt, pad)
            images_norm, mask_final = post_optim_normalization(images.cpu().detach().numpy())
            full_path = os.path.join(base_folder, branch_folder, this_folder)
            full_path = os.path.normpath(full_path)
            show_animations_in_notebook(images_norm, full_path, generation_level=cc, grayScale=var.grayScale)
            activation_layer = get_actv(model, outputs, x.clone().detach(), var.actv_layers[0], mask, transforms)
        actv_layer_avg = torch.mean(activation_layer, dim=(-2, -1))
        stimulus_list.append(actv_layer_avg.clone().detach())
        cc = cc + 1
    else:
        # Paper setting: keep the raw optimized pair
        stimulus_list.append(x.clone().detach())

    loss_at_saved_images.append(loss)

    if var.backward is False:
        return stimulus_list, this_loss, loss_at_saved_images, classifiers, transforms

    # ==================================================================
    # "Backward" scheme (NOT used in the paper; var.backward = False).
    # Starting from the optimized pair, the classifier term is reversed to
    # move the stimuli back toward each other. Kept for completeness.
    # ==================================================================
    optimizer_backward = torch.optim.Adam([x], lr=var.back_lr)
    if var.scheduled_back_lr == True:
        scheduler_backward = torch.optim.lr_scheduler.StepLR(optimizer_backward, step_size=var.step_size_back, gamma=var.gamma_back)

    num_backward_steps = var.backward_iteration

    for i in tqdm(range(num_backward_steps)):

        decay = decay_factor_logistic(i,
                                      var.frw_iterations,
                                      min_value=0.0,
                                      steepness=var.frw_logistic_func_steepness,
                                      midpoint=var.frw_logistic_func_midpoint)
        param_jitter_2 = int(np.round(13 * decay))
        param_jitter_1 = int(param_jitter_2 / 2)

        if param_jitter_2 < 2:
            transforms = [random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0)]
        elif param_jitter_1 < 2:
            transforms = [random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0),
                          jitter(param_jitter_2, decay=1.0)]
        else:
            transforms = [jitter(param_jitter_1, decay=1.0),
                          random_scale([var.frw_base_scale * (1 + (i - 25) / 250.0) for i in range(51)], decay=1.0),
                          jitter(param_jitter_2, decay=1.0)]

        x_ = transform(x * mask, transforms)
        x_ = x_.to(device)
        _ = model(clip(x_)).to(device)

        # Switch the sign for the layer whose separation is being reduced
        switch_sign = []
        CEoutput_layers = []
        for index in range(len(layers)):
            cz, cy, cx = outputs[layers[index]].shape[2] // 2, outputs[layers[index]].shape[3] // 2, outputs[layers[index]].shape[4] // 2

            LCinput = classifiers[index].linclassifier(outputs[layers[index]][:, :, :, cy, cx].reshape((var.nex_in, -1)))

            if var.loss_func == 'CrossEntropyLoss':
                CEoutput = - classifiers[index].CEloss(LCinput, classifiers[index].linlabels)

            elif var.loss_func == 'BCELoss':
                sig_func = nn.Sigmoid()

                LCinput = LCinput.squeeze()
                classifiers[index].linlabels = classifiers[index].linlabels.float()
                CEoutput = - classifiers[index].CEloss(sig_func(LCinput), classifiers[index].linlabels)

            CEoutput_layers.append(CEoutput)
            if classifiers[index].control_actv == +1:
                switch_sign.append(-1)
            elif classifiers[index].control_actv == -1:
                switch_sign.append(+1)
            else:
                switch_sign.append(0)

        assert x.shape[1] == 3

        xp = F.conv3d(x, filt, groups=3, padding=(0, pad, pad)) * mask
        x_averaged = clip(xp).mean(axis=1).reshape(x.shape[0], x.shape[2], -1)
        x_staggered_flipped = torch.flip(torch.roll(xp, 1, 0), [2])
        x_mixed = clip(.707 * (xp + x_staggered_flipped)).mean(1).reshape(x.shape[0], x.shape[2], -1)

        this_loss.append([float(CEoutput_layers[0]), float(CEoutput_layers[1]), float(CEoutput_layers[2]), float(CEoutput_layers[3]),
                          float(var.mean_penalty * ((clip(x)) ** 2).mean()),
                          float(var.color_penalty * (clip(xp).mean(axis=3).mean(axis=3) ** 2).mean()),
                          float(var.contrast_penalty * (((x_averaged ** 2).mean(axis=2) - target_contrast) ** 2).mean()),
                          float(var.mixed_contrast_penalty * (((x_mixed ** 2).mean(axis=2) - target_mixed_contrast) ** 2).mean())])

        loss = (CEoutput_layers[0] * CEcoeff * classifiers[0].control_actv * switch_sign[0] +
                CEoutput_layers[1] * CEcoeff * classifiers[1].control_actv * switch_sign[1] +
                CEoutput_layers[2] * CEcoeff * classifiers[2].control_actv * switch_sign[2] +
                CEoutput_layers[3] * CEcoeff * classifiers[3].control_actv * switch_sign[3] +
                var.mean_penalty * ((clip(x)) ** 2).mean() +
                var.color_penalty * (clip(xp).mean(axis=3).mean(axis=3) ** 2).mean() +
                var.contrast_penalty * (((x_averaged ** 2).mean(axis=2) - target_contrast) ** 2).mean() +
                var.mixed_contrast_penalty * (((x_mixed ** 2).mean(axis=2) - target_mixed_contrast) ** 2).mean())

        optimizer_backward.zero_grad()
        (loss).backward()
        optimizer_backward.step()
        if var.scheduled_back_lr == True:
            scheduler_backward.step()

        # Save every `generation_interval` iterations (starting from i + 1 so
        # the pair saved before the loop is not duplicated)
        if (var.generation_interval != None) and ((i + 1) % var.generation_interval == 0):

            if var.save_stim_atIter == True:
                if var.save_gif == True:
                    print("saving stim")
                    images = post_optim_manipulations(x.clone().detach(), filt, pad)
                    images_norm, mask_final = post_optim_normalization(images.cpu().detach().numpy())
                    full_path = os.path.join(base_folder, branch_folder, this_folder)
                    full_path = os.path.normpath(full_path)
                    show_animations_in_notebook(images_norm, full_path, generation_level=cc, grayScale=var.grayScale)
                activation_layer = get_actv(model, outputs, x.clone().detach(), var.actv_layers[0], mask, transforms)
                actv_layer_avg = torch.mean(activation_layer, dim=(-2, -1))
                stimulus_list.append(actv_layer_avg.clone().detach())
                cc = cc + 1
            else:
                stimulus_list.append(x.clone().detach())

            loss_at_saved_images.append(loss)

    return stimulus_list, this_loss, loss_at_saved_images, classifiers, transforms
