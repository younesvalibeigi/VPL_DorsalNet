"""
Configuration for the ANN readout modeling experiment (paper Section 2.4).

The values below are the ones used for the paper's run
(10 000 epochs, seven readout architectures run in this order).
"""

import os

import torch

from func_transforms import random_scale

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

# Repository root (one level above this file)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class beh_Model_Variables:
    def __init__(self):
        # Stimulus categories (target layers of the synthesized stimuli)
        self.layers = ['res0', 'res2']
        self.colors = ['b', 'g', 'r', 'orange']

        # ------------------------------------------------------------------
        # Repetitions / stimulus sets
        # repeatNotVar = True: repeat the experiment `repeat` times on the
        # stimulus set variation_array[0].
        # ------------------------------------------------------------------
        self.repeatNotVar = True
        self.repeat = 1
        self.variation_array = [str(i) for i in range(1, 1 + 1)]  # ['1'] -> folder suffix "_var_1"

        # ------------------------------------------------------------------
        # Task and data location
        # 'binary': classify single stimuli into their category of origin
        # (the paper's ANN task). 'same_different' is an alternative pair task.
        # ------------------------------------------------------------------
        self.task = 'binary'
        if self.task == 'same_different':
            self.num_cond = 4
        elif self.task == 'binary':
            self.num_cond = 2
        self.raw_data = True              # load the raw optimized tensors (all_stimuli.pt), not the GIFs
        self.inhibitOtherLayers = False   # must match the stimulus-generation setting
        # Folder written by Section 2 (stimulus_generation/Model_Variables.py)
        self.main_address = os.path.join(_REPO_ROOT, 'data', 'generated_stimuli')

        self.save_results = False
        self.save_data_address = os.path.join(_REPO_ROOT, 'results', 'ann_readout', 'figures')
        if not os.path.exists(self.save_data_address):
            os.makedirs(self.save_data_address)

        # Radius (px) of the stimulus region: circular mask applied to the
        # stimuli, size of the square moved for the location test, and radius
        # of the circular noise patch added on every pass
        self.crop_radius = 66
        self.spatial_size = None          # keep the 164 x 164 canvas
        # Location test (Section 2.4.3): stimuli are trained at the centre and
        # tested after moving them toward the bottom-right of the 164 x 164 grid
        self.position_train = 'center'
        self.position_loc_test = 'bottom-right'
        self.shift_px = None

        self.classifier_model_type = 'singleLayerReadout'

        # ------------------------------------------------------------------
        # Readout architectures (Section 2.4.1), run in this order. The order
        # matters for exact reproduction because the random state carries
        # over from one readout to the next.
        #   'res0'..'res3' : one residual layer (352 features)
        #   'modelOutput'  : DorsalNet's final output
        #   'All'          : res0-res3 concatenated (1408 features)
        #   'singleLayer'  : stimulus-driven readout (res0 for res0 stimuli,
        #                    res2 for res2 stimuli)
        # ------------------------------------------------------------------
        self.readout = 'All'
        self.readout_arr = ['res0', 'res1', 'res2', 'res3', 'modelOutput', 'All', 'singleLayer']

        # Ablation (not used in the paper; requires readout = 'All'):
        # 'res0'..'res3' zero that layer's features, 'layer_tunedStimulus'
        # zeroes the layer the stimulus was made for, a number in (0, 1)
        # zeroes that fraction of all features, False = no ablation.
        self.ablate = False
        self.ablate_arr = [False]

        # Simulated microsaccades (not used in the paper)
        self.microsaccade = False
        self.microsaccade_shift_px = 10

        # Early stopping at a performance threshold (disabled)
        self.num_epochs_res0 = None       # keep None
        self.threshold_performance = None

        # ------------------------------------------------------------------
        # Training parameters (Section 2.4.2)
        # ------------------------------------------------------------------
        self.num_augmentations = 0
        self.num_epochs = 10000
        self.lr = 0.00001                 # Adam learning rate
        # levels = 1 + fromHardToEasy = False -> the training set is the fully
        # optimized target pair only
        self.fromHardToEasy = False
        self.noiseLevels_generateWithOrgStim = False
        self.levels = 1
        self.batch_size = int(self.levels * self.num_cond)
        self.dropout = None
        self.differentCondWeight = None
        # SD of the grayscale Gaussian noise added inside the circular patch
        # on every training and evaluation pass
        self.guassian_noise_std = 0.01

        # Learning-rate schedule (off)
        self.sch_lr = False
        self.gamma = None
        self.step_size = 50

        # Print the training loss every `printEpoch` epochs
        self.printEpoch = 100

        # Stochastic random-rescaling transform (same as in stimulus
        # synthesis, Section 2.2.3): resize by 0.25 * [0.9, 1.1] and zero-pad
        # back to 164 x 164 before DorsalNet.
        self.transforms = [random_scale([0.25 * (1 + (i - 25) / 250.0) for i in range(51)])]

        # Marker
        self.printMsg = 'Before here are general variables for the run, after here are variables specific for this file'
