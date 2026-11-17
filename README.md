# Hierarchical control of visual perceptual learning with model-synthesized stimuli

Code for **Valibeigi, Pandey, Pack & Bakhtiari**, *Hierarchical control of visual perceptual learning with model-synthesized stimuli*.

We use **DorsalNet** (Mineault et al., 2021), a hierarchical neural network model of the primate dorsal stream, to synthesize motion stimuli that are maximally discernible only by an early (`res0`, V1/MT-like) or a late (`res2`, MST-like) processing stage. We then train linear readouts of the frozen model on the same task the observers performed, and compare readout architectures by how well they reproduce the human pattern of location and stimulus-category specificity.

This repository contains:

1. **Stimulus synthesis** (paper Section 2.2): PCA readout directions, optimization of stimulus pairs that target a single DorsalNet layer, graded "optimization levels", and post-processing into display-ready GIFs.
2. **ANN readout modeling** (paper Section 2.4): linear classifiers on the frozen DorsalNet, trained on the synthesized stimuli and tested for location and stimulus-category transfer, for seven readout architectures.
3. **Visualization** of the stimuli, learning curves and specificity indices.

Everything is run from a single notebook, [`DorsalNet_VPL.ipynb`](DorsalNet_VPL.ipynb).

---

## Repository structure

```
VPL_DorsalNet/
├── DorsalNet_VPL.ipynb            # main notebook (Sections 1-4)
├── stimulus_generation/           # Section 2: stimulus synthesis
│   ├── Model_Variables.py         #   all synthesis settings
│   ├── generate_x.py              #   initial stimulus pair, mask, Gaussian filter
│   ├── classifier_objects.py      #   fixed PCA-based linear readout classifiers
│   ├── optim_stim_multiObj.py     #   target-pair optimization
│   ├── generate_noise_levels.py   #   graded optimization levels
│   ├── func_transforms.py         #   random jitter / rescaling transforms
│   ├── post_stim_anal.py          #   smoothing, masking, normalization, GIF export
│   ├── general_func.py            #   layer activations, saving settings
│   ├── threeD_pca.py              #   3D PCA plots of the levels
│   ├── plotting_func.py           #   loss plots
│   └── presentHTML.py             #   HTML summary pages
├── ann_readout/                   # Section 3: ANN readout modeling
│   ├── beh_Model_Variables.py     #   all readout-experiment settings
│   ├── load_all_gen_imgs.py       #   load the res0 / res2 stimulus sets
│   ├── load_raw_data.py           #   read all_stimuli.pt
│   ├── prepare_model_in_out.py    #   training and location-transfer sets
│   ├── beh_classifier_model.py    #   central-unit linear readout classifier
│   ├── simultaneous_train_test_model.py  # training + per-epoch transfer tests
│   ├── plotting_tools.py          #   summary figures
│   ├── stimulus_dataset_utils.py  #   noise patch, location shift, helpers
│   ├── stimulus_transforms.py     #   transform helpers (clip)
│   └── filters3d.py               #   circular mask and 3D filters
├── pca_components/                # PCA readout directions (pca_components_res*.pt)
├── requirements.txt               # extra Python packages
├── LICENSE
└── README.md
```

Created when the notebook runs (not tracked):

```
yhit/          DorsalNet code (cloned), includes the lucentpatch package
lucent/        lucent library (cloned and installed)
checkpoints/   DorsalNet weights (downloaded)
data/generated_stimuli/   synthesized stimuli (Section 2)
results/ann_readout/      readout results (Section 3)
results/figures/          figures (Section 4)
```

---

## Installation

A CUDA-capable GPU is strongly recommended.

```bash
git clone https://github.com/younesvalibeigi/VPL_DorsalNet.git
cd VPL_DorsalNet
conda create -n dorsalnet python
conda activate dorsalnet
pip install jupyter gdown
jupyter notebook DorsalNet_VPL.ipynb
```

Section 1 of the notebook then does the rest:

1. clones DorsalNet ([`your-head-is-there-to-move-you-around`](https://github.com/patrickmineault/your-head-is-there-to-move-you-around)) into `yhit/`;
2. downloads the DorsalNet checkpoints into `checkpoints/`;
3. points `yhit/paths.py` to `checkpoints/`;
4. installs the dependencies:
   ```bash
   conda install --file yhit/requirements.txt -y
   git clone https://github.com/greentfrapp/lucent.git
   cd lucent && pip install . && cd ..
   pip install -r requirements.txt
   ```
5. loads the frozen DorsalNet (`airsim_04`) and attaches forward hooks to `s1` and `res0`–`res3`.

Steps 1–4 only need to be run once.

---

## Running the pipeline

Open `DorsalNet_VPL.ipynb` from the repository root and run the sections in order.

### Section 2: stimulus generation

| Step | Paper | What happens |
|---|---|---|
| 2.1 PCA readout directions *(optional)* | 2.2.2 | PCA of each layer's activations to the AirSim training videos. The resulting files are provided in `pca_components/`, so this step can be skipped (`RUN_PCA = False`). |
| 2.2 Load PCA components | 2.2.2 | PC2 of each layer becomes the fixed weight vector of that layer's readout classifier. |
| 2.3 Synthesize stimuli | 2.2.3–2.2.5 | Target-pair optimization, graded optimization levels, post-processing and GIF export, loss plots, 3D PCA plots and HTML summaries. |

Settings: `stimulus_generation/Model_Variables.py`.
Output: `data/generated_stimuli/<layer>_Inhb_False_classifierPCA2__nex_2_var_1/`, with `all_stimuli.pt` of shape `[26, 2, 3, 11, 164, 164]` (index 0 = optimized target pair, 1–25 = optimization levels), one GIF per stimulus and level, and a `present.html` overview.

### Section 3: ANN readout modeling

For each of the seven readout architectures (`res0`, `res1`, `res2`, `res3`, `modelOutput`, `All`, `singleLayer`), one classifier is trained on res0 stimuli and one on res2 stimuli. After every training step each classifier is evaluated on the training stimuli, on the same stimuli at a shifted location, and on the other stimulus category.

| Readout | Reads from | Hypothesis |
|---|---|---|
| `res0`, `res1` | one early layer | early-stage reweighting |
| `res2`, `res3` | one late layer | late-stage reweighting |
| `modelOutput` | DorsalNet's output | — |
| `All` | res0–res3 concatenated | learning picks the layer(s) |
| `singleLayer` | res0 for res0 stimuli, res2 for res2 stimuli | stimulus-driven (adaptive-stage) reweighting |

Settings: `ann_readout/beh_Model_Variables.py` (10 000 epochs, Adam lr = 1e-5, zero-initialized weights).
Output: `results/ann_readout/run_<timestamp>_epochs_10000/beh_results_readout_<readout>_ablate_False.pkl`.

### Section 4: visualization

* Synthesized stimuli and their 3D PCA plots.
* Training and location-test performance of one readout.
* Location and stimulus-category specificity of every readout.

---

## Reproducibility

* The global seed (Python `random`, NumPy, PyTorch CPU/GPU) is set to **42** before stimulus generation and again before the ANN readout experiment; cuDNN runs in deterministic mode.
* Keep `layers_actv = ['res0', 'res1', 'res2', 'res3']` in `Model_Variables.py`. Only res0 and res2 stimuli are used, but generating all four layers in this order keeps the random-number stream, and therefore the res2 stimuli, identical to the paper.
* Keep the order of `readout_arr` in `beh_Model_Variables.py`; the random state carries over from one readout to the next.
* Some GPU operations are not bit-for-bit deterministic across hardware and library versions, so small numerical differences are possible on a different machine.

---

## Acknowledgments

DorsalNet and the `lucentpatch` package come from Mineault et al. (2021), [`your-head-is-there-to-move-you-around`](https://github.com/patrickmineault/your-head-is-there-to-move-you-around). Feature-visualization utilities come from [lucent](https://github.com/greentfrapp/lucent).

> Mineault, P., Bakhtiari, S., Richards, B., & Pack, C. (2021). Your head is there to move you around: Goal-driven models of the primate dorsal pathway. *Advances in Neural Information Processing Systems*, 34.

## Citation

If you use this code, please cite:

> Valibeigi, Y., Pandey, A., Pack, C. C., & Bakhtiari, S. *Hierarchical control of visual perceptual learning with model-synthesized stimuli.*

## License

Released under the [MIT License](LICENSE).
