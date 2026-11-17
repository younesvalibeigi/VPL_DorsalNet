"""
stimulus_generation
===================

Model-based synthesis of motion stimuli that target a single processing
stage of DorsalNet (paper Section 2.2, "Stimulus Synthesis").

Pipeline (driven from ``DorsalNet_VPL.ipynb``, Section 2):

1. ``generate_x``            - random initial stimulus pair, spatial mask and
                               Gaussian filter (Section 2.2.3).
2. ``optim_stim_multiObj``   - optimizes the pair to be maximally separable
                               along a fixed PCA readout direction of the
                               target layer (Sections 2.2.2-2.2.3).
3. ``generate_noise_levels`` - graded series of stimulus pairs ("optimization
                               levels") that move from noise toward the
                               optimized target pair (Section 2.2.5).
4. ``post_stim_anal``        - display post-processing: smoothing, masking,
                               up-sampling, contrast/luminance normalization
                               and GIF export (Section 2.2.4).
5. ``threeD_pca`` / ``presentHTML`` / ``plotting_func`` - diagnostic figures
                               and HTML summary pages.

The modules use flat imports (``from func_transforms import ...``), so this
folder is added to ``sys.path`` by the notebook rather than imported as a
package.
"""
