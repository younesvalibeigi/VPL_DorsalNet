"""
Load the raw synthesized stimuli (``all_stimuli.pt``) written in Section 2.

``all_stimuli.pt`` has shape [n_pairs, 2, 3, 11, 164, 164]:
index 0 is the fully optimized target pair, indices 1.. are the graded
series from generate_noise_levels (noise -> target-like).
"""

import os

import torch

# Map the saved tensors to the GPU when available, otherwise to the CPU
_MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_raw_data(var, stim_address, task='same-different', layer_stim_test=None, device='None'):
    """Same-different version: every pair gives 4 trials
    (AB, BA -> label 0 'different'; AA, BB -> label 1 'same')."""
    file_path = os.path.join(stim_address, 'all_stimuli.pt')
    all_stimuli = torch.load(file_path, map_location=_MAP_LOCATION)  # [26, 2, 3, 11, 164, 164]
    num_pairs = all_stimuli.shape[0]
    num_cond = var.num_cond  # 4 for same-different

    beh_X = torch.zeros((num_pairs * num_cond, 2, 3, 11, 164, 164)).to(device)
    beh_Y = torch.zeros((num_pairs * num_cond, 1)).to(device)

    for i in range(num_pairs):
        stim1 = all_stimuli[i, 0]  # [3, 11, 164, 164]
        stim2 = all_stimuli[i, 1]

        # different, different, same, same -> labels 0, 0, 1, 1
        beh_X[i * num_cond + 0] = torch.stack((stim1, stim2), dim=0)
        beh_Y[i * num_cond + 0] = 0

        beh_X[i * num_cond + 1] = torch.stack((stim2, stim1), dim=0)
        beh_Y[i * num_cond + 1] = 0

        beh_X[i * num_cond + 2] = torch.stack((stim1, stim1), dim=0)
        beh_Y[i * num_cond + 2] = 1

        beh_X[i * num_cond + 3] = torch.stack((stim2, stim2), dim=0)
        beh_Y[i * num_cond + 3] = 1

    beh_X = beh_X.to(device)
    beh_Y = beh_Y.to(device)

    return beh_X, beh_Y


def load_raw_data_binary(var, stim_address, task='same-different', layer_stim_test=None, device='None'):
    """Binary version used in the paper: each stimulus is one sample,
    labelled by the member of the pair it belongs to (stim1 -> 0, stim2 -> 1).

    Returns beh_X [n_pairs * 2, 3, 11, 164, 164] and beh_Y [n_pairs * 2, 1],
    ordered as [pair0_stim1, pair0_stim2, pair1_stim1, ...].
    """
    file_path = os.path.join(stim_address, 'all_stimuli.pt')
    all_stimuli = torch.load(file_path, map_location=_MAP_LOCATION)  # [26, 2, 3, 11, 164, 164]
    num_pairs = all_stimuli.shape[0]
    num_cond = 2  # binary task

    beh_X = torch.zeros((num_pairs * num_cond, 3, 11, 164, 164)).to(device)
    beh_Y = torch.zeros((num_pairs * num_cond, 1)).to(device)

    for i in range(num_pairs):
        stim1 = all_stimuli[i, 0]  # [3, 11, 164, 164]
        stim2 = all_stimuli[i, 1]

        beh_X[i * num_cond + 0] = stim1
        beh_Y[i * num_cond + 0] = 0

        beh_X[i * num_cond + 1] = stim2
        beh_Y[i * num_cond + 1] = 1

    beh_X = beh_X.to(device)
    beh_Y = beh_Y.to(device)

    return beh_X, beh_Y
