"""
Configuration for the stimulus-synthesis stage (paper Section 2.2).

Every hyper-parameter read by ``optim_stim_multiObj`` (Section 2.2.3) and
``generate_noise_levels`` (Section 2.2.5) is defined here. The values below
are the ones used to produce the stimuli reported in the paper.
"""

import os

import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

# Repository root (one level above this file); all outputs are written below it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Model_Variables:
    def __init__(self):
        # ------------------------------------------------------------------
        # Output folders
        # Stimuli for each target layer are written to
        #   <base_folder>/<branch_folder>/<layer>_Inhb_<..>_classifierPCA<..>__nex_<..>_var_<..>/
        # The ANN readout experiment (ann_readout/beh_Model_Variables.py)
        # reads them back from the same location.
        # ------------------------------------------------------------------
        self.base_folder = os.path.join(_REPO_ROOT, 'data')
        self.branch_folder = 'generated_stimuli'

        # Save every synthesized stimulus as a GIF (display-ready, Section 2.2.4)
        self.save_gif = True

        # Number of independent stimulus sets to generate per target layer
        self.diversity_variation = 1

        # Randomness and seed settings. Global seeding is done in the notebook
        # (seed = 42) right before the generation loop; these flags are only
        # recorded in variables.txt for bookkeeping.
        self.randomness = False
        self.seed_value = False
        self.seed = self.seed_value if self.randomness else 'False'

        # ------------------------------------------------------------------
        # Optimization-level generation (Section 2.2.5)
        # After the target pair is optimized, a second pair is started from
        # full noise and pulled toward the target pair's activations. Saving
        # it every `gen_nosie_interval` iterations yields a graded series of
        # pairs, from noisy/hard to target-like/easy.
        # ------------------------------------------------------------------
        self.gen_noise_case = True        # generate the graded series
        self.save_stim_atIter = False
        self.backward = False             # alternative "move backward" scheme (not used in the paper)

        self.gen_noise_lr = 0.001         # Adam learning rate for the graded series
        self.gen_noise_iter = 1200        # total iterations
        self.gen_noise_sch_lr = False     # no learning-rate schedule
        self.start_scheduled_gen_noise_lr = 0
        self.gen_nosie_interval = 50      # save the evolving pair every 50 iterations
        self.gen_step_size = 100
        self.gamma_noise = 1.6
        self.gen_noise_save_stim = True
        # The transforms reached at the end of the target-pair optimization are
        # carried over unchanged (see generate_noise_levels).
        self.gen_noise_transforms_outsideLoop = True
        self.gen_noise_base_scale = 0.25
        self.gen_noise_logistic_func_steepness = 20
        self.gen_noise_logistic_func_midpoint = 0.25

        # ------------------------------------------------------------------
        # Model / classifier settings (Section 2.2.2)
        # ------------------------------------------------------------------
        self.grayScale = True             # GIFs are converted to grayscale so observers rely on motion, not colour
        self.loss_func = 'BCELoss'        # one-output linear classifier + sigmoid

        # Target layers. Each layer is processed in turn; only res0 (early) and
        # res2 (late) stimuli are used in the experiments, but all four layers
        # are kept so the random-number stream (and therefore the res2
        # stimuli) is identical to the run reported in the paper.
        self.layers_actv = ['res0', 'res1', 'res2', 'res3']
        self.all_layers = ['res0', 'res1', 'res2', 'res3']
        # No explicit inhibition of the non-target layers: their classifier
        # terms get weight 0 in the objective (control_actv = 0).
        self.inhibit_otherLayers_arr = [False]
        self.inhibit_top_layers_arr = [False]

        # Fixed readout direction: the 2nd principal component (PC2) of each
        # layer's activations to the AirSim training videos. PC2 gave the
        # clearest, most coherent motion among the candidates compared.
        self.weights = 'pca'              # 'pca', 'rand' or 'ones'
        self.pca_num_arr = [2]
        # Stimuli per optimization batch: 2 = one pair (labels 0 and 1)
        self.nex_in_arr = [2]

        # Read out only the central spatial unit of each layer (32 channels x
        # 11 frames = 352 features). This confines the synthesized motion to
        # the centre of the frame.
        self.central_pixel = True

        # ------------------------------------------------------------------
        # Target-pair optimization (Section 2.2.3)
        # Adam, lr = 0.1, 2000 iterations; after iteration 1200 the learning
        # rate is multiplied by 0.5 every 100 scheduler steps.
        # ------------------------------------------------------------------
        self.frw_lr = 0.1
        self.scheduled_frw_lr = True
        self.frw_iterations = 2000
        self.start_scheduled_frw_lr = 1200
        self.step_size_frw = 100
        self.frw_gamma = 0.5
        # Random rescaling: each iteration the stimulus is resized by a factor
        # drawn from 0.25 * [0.9, 1.1] (i.e. 0.225-0.275) and zero-padded back
        # to 164 x 164 before entering DorsalNet.
        self.frw_base_scale = 0.25
        # Logistic decay of the jitter magnitude (largest early, zero late)
        self.frw_logistic_func_steepness = 20
        self.frw_logistic_func_midpoint = 0.25

        # Gradient clipping (tested; left off)
        self.clip_frw = False
        self.max_norm_frw = 1.0

        # ------------------------------------------------------------------
        # "Backward" scheme (only used when self.backward is True; not used
        # in the paper)
        # ------------------------------------------------------------------
        self.backward_iteration = 3000
        self.generation_interval = 1000
        self.scheduled_back_lr = False
        self.gamma_back = 1.6
        self.step_size_back = self.generation_interval
        self.back_lr = 0.001
        self.rnd_seed_ar = [self.seed_value]

        # ------------------------------------------------------------------
        # Auxiliary penalty weights (Section 2.2.3, terms i-iv)
        #   mean_penalty           (i)   mean-squared pixel value -> neutral gray
        #   color_penalty          (ii)  per-channel colour deviation -> no colour cast
        #   contrast_penalty       (iii) per-frame contrast of the smoothed stimulus -> target
        #   mixed_contrast_penalty (iv)  contrast of the superimposed pair -> second target
        # ------------------------------------------------------------------
        self.mean_penalty = 1.5
        self.contrast_penalty = 50.0
        self.color_penalty = 1000.0
        self.mixed_contrast_penalty = 50.0

        # Keep classifier weights frozen (only the stimulus is optimized)
        self.requireGrad = False

        # Marker written to variables.txt
        self.printMsg = 'Before here are general varialbes for the run, after here are variables specific for this file'
