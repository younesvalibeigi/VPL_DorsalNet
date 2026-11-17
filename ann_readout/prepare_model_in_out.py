"""
Build the training and location-test sets for the ANN readout experiment.

Paper reference: Sections 2.4.2 (Training Procedure) and 2.4.3 (Transfer
Testing).
"""

import numpy as np
import torch

from stimulus_transforms import StimulusTransforms
from stimulus_dataset_utils import StimulusDatasetUtils
from filters3d import Filters3D


def prepare_model_in_out(var, device, beh_X_res0, beh_Y_res0, beh_X_res2, beh_Y_res2):
    """Select the training stimuli, mask/clip them, and build the
    location-transfer set.

    Steps
    -----
    1. Roll the loaded arrays by one pair so the fully optimized target pair
       (stored first in all_stimuli.pt) moves to the end, after the graded
       series. With ``levels = 1`` and ``fromHardToEasy = False`` the last
       level, i.e. the target pair, is selected as the training set.
    2. Apply the soft circular mask (radius ``crop_radius`` = 66 px) and clip
       values to [-2, 2] (input format used during stimulus synthesis).
    3. Location test: move the central 132 x 132 square toward the
       bottom-right of the 164 x 164 grid (uniform background value 0).

    Returns
    -------
    xRes0_train, yRes0, xRes0_loc_test, xRes2_train, yRes2, xRes2_loc_test
    """
    ST = StimulusTransforms(device)
    f = Filters3D(device=device)
    SDU = StimulusDatasetUtils(device=device)

    # --------------------------------------------------------------
    # Indices of the levels used for training
    # --------------------------------------------------------------
    if var.task == 'same_different':
        num_cond = 4
        if var.fromHardToEasy:
            even_levels = torch.arange(0, var.levels)
            level_indices = torch.cat([torch.arange(l * num_cond, (l + 1) * num_cond) for l in even_levels])
        else:
            even_levels = torch.arange(26 - var.levels, 26)
            level_indices = torch.cat([torch.arange(l * num_cond, (l + 1) * num_cond) for l in even_levels])

    elif var.task == 'binary':
        num_cond = 2
        if var.fromHardToEasy:
            even_levels = torch.arange(0, var.levels)
            level_indices = torch.cat([torch.arange(l * num_cond, (l + 1) * num_cond) for l in even_levels])
        else:
            # Paper setting: the easiest `levels` levels (levels = 1 -> the
            # fully optimized target pair, after the roll below)
            even_levels = torch.arange(26 - var.levels, 26)
            level_indices = torch.cat([torch.arange(l * num_cond, (l + 1) * num_cond) for l in even_levels])

    # Move the target pair (index 0) to the end so levels are ordered
    # noise -> ... -> target
    beh_X_res0_corrected = torch.roll(beh_X_res0, shifts=-2, dims=0)
    beh_Y_res0_corrected = torch.roll(beh_Y_res0, shifts=-2, dims=0)
    beh_X_res2_corrected = torch.roll(beh_X_res2, shifts=-2, dims=0)
    beh_Y_res2_corrected = torch.roll(beh_Y_res2, shifts=-2, dims=0)

    # Optional (not used in the paper): build difficulty levels by adding
    # noise to the target pair instead of using the synthesized levels
    if var.noiseLevels_generateWithOrgStim:
        beh_X_res0_corrected_temp = beh_X_res0_corrected
        beh_X_res2_corrected_temp = beh_X_res2_corrected
        num_stim = beh_X_res0_corrected.shape[0]
        for step, i_pair in enumerate(range(0, num_stim, 2)):
            num_steps = num_stim // 2  # 26
            r = (0.01) ** (1 / 23)
            linear_values = np.linspace(2, 0, num_steps - 1)  # 25 values
            if step == 0:
                std_value = 10.0
            elif step == num_steps - 1:
                std_value = 0.0
            else:
                std_value = linear_values[step - 1]

            stims_res0 = beh_X_res0_corrected[-2:]
            stims_res2 = beh_X_res2_corrected[-2:]
            stims_res0 = SDU.add_bw_gaussian_circle(stims_res0, mean=0.0, std=std_value, radius=var.crop_radius)
            stims_res2 = SDU.add_bw_gaussian_circle(stims_res2, mean=0.0, std=std_value, radius=var.crop_radius)
            beh_X_res0_corrected_temp[i_pair:i_pair + 2] = stims_res0
            beh_X_res2_corrected_temp[i_pair:i_pair + 2] = stims_res2

        beh_X_res0_corrected = beh_X_res0_corrected_temp
        beh_X_res2_corrected = beh_X_res2_corrected_temp

    # --------------- Preprocess res0 stimuli ----------------- #
    if var.task == 'same_different':
        beh_X_comb_res0 = SDU.combine_input_pairs(beh_X_res0_corrected[level_indices])
    elif var.task == 'binary':
        beh_X_comb_res0 = beh_X_res0_corrected[level_indices]
    masked_data_res0 = ST.clip(f.apply_mask(beh_X_comb_res0, radius=var.crop_radius))
    if var.position_train == 'center':
        xRes0_train = masked_data_res0
    else:
        xRes0_train = SDU.move_center_square_to_corner(masked_data_res0, edge=var.crop_radius * 2,
                                                       gray_value=0.0, position=var.position_train, output_size=var.spatial_size,
                                                       shift_px=var.shift_px)

    # Location-transfer set
    if var.position_loc_test == 'rotate':
        xRes0_loc_test = SDU.rotate_90(masked_data_res0)
    else:
        xRes0_loc_test = SDU.move_center_square_to_corner(masked_data_res0, edge=var.crop_radius * 2,
                                                          gray_value=0.0, position=var.position_loc_test, output_size=var.spatial_size,
                                                          shift_px=var.shift_px)
    yRes0 = beh_Y_res0_corrected[level_indices]

    # --------------- Preprocess res2 stimuli ----------------- #
    if var.task == 'same_different':
        beh_X_comb_res2 = SDU.combine_input_pairs(beh_X_res2_corrected[level_indices])
    elif var.task == 'binary':
        beh_X_comb_res2 = beh_X_res2_corrected[level_indices]
    masked_data_res2 = ST.clip(f.apply_mask(beh_X_comb_res2, radius=var.crop_radius))
    if var.position_train == 'center':
        xRes2_train = masked_data_res2
    else:
        xRes2_train = SDU.move_center_square_to_corner(masked_data_res2, edge=var.crop_radius * 2,
                                                       gray_value=0.0, position=var.position_train, output_size=var.spatial_size)

    # Location-transfer set
    if var.position_loc_test == 'rotate':
        xRes2_loc_test = SDU.rotate_90(masked_data_res2)
    else:
        xRes2_loc_test = SDU.move_center_square_to_corner(masked_data_res2, edge=var.crop_radius * 2,
                                                          gray_value=0.0, position=var.position_loc_test, output_size=var.spatial_size)
    yRes2 = beh_Y_res2_corrected[level_indices]

    # --------------- Return prepared data ----------------- #
    xRes0_train = xRes0_train.to(device)
    xRes0_loc_test = xRes0_loc_test.to(device)
    yRes0 = yRes0.to(device)
    xRes2_train = xRes2_train.to(device)
    xRes2_loc_test = xRes2_loc_test.to(device)
    yRes2 = yRes2.to(device)

    return xRes0_train, yRes0, xRes0_loc_test, xRes2_train, yRes2, xRes2_loc_test
