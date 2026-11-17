"""
ann_readout
===========

ANN readout modeling experiment (paper Section 2.4). In the code this part is
often called "beh" (behaviour), because the classifiers model the observers'
behaviour on the trained task.

A linear classifier with a sigmoid output reads activations of the FROZEN
DorsalNet model and learns to classify individual synthesized stimuli into
their category of origin (label 0 / 1). Different readout architectures
(which layer(s) are read out) implement different hypotheses about where
reweighting acts during perceptual learning:

    'res0', 'res1'  -> early-stage reweighting
    'res2', 'res3'  -> late-stage reweighting
    'All'           -> all four residual layers concatenated
    'singleLayer'   -> stimulus-driven (adaptive-stage) readout: res0 for res0
                       stimuli, res2 for res2 stimuli
    'modelOutput'   -> DorsalNet's final output

After training, each classifier is tested on (i) the training set,
(ii) the same stimuli shifted to a new location and (iii) the other stimulus
category, mirroring the human transfer tests.

Pipeline (driven from ``DorsalNet_VPL.ipynb``, Section 3):

    beh_Model_Variables            - all settings
    load_all_gen_imgs / load_raw_data - read the stimuli from Section 2
    prepare_model_in_out           - masking, clipping, location-test set
    beh_classifier_model           - Classifier_centralpixel_binary
    simultaneous_train_test_model  - training + per-epoch evaluation
    plotting_tools                 - LossPlotter summary figures

The modules use flat imports, so this folder (and ``stimulus_generation``)
is added to ``sys.path`` by the notebook.
"""
