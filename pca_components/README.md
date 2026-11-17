# PCA readout directions

Principal components of each DorsalNet residual layer's activations to the AirSim self-motion videos (paper Section 2.2.2). They are loaded in Section 2.2 of the notebook and can be recomputed in Section 2.1.

Expected files:

| File | Shape | Content |
|---|---|---|
| `pca_components_res0.pt` | `[20, 352]` | 20 components of res0 (32 channels × 11 frames, spatially averaged) |
| `pca_components_res1.pt` | `[20, 352]` | same for res1 |
| `pca_components_res2.pt` | `[20, 352]` | same for res2 |
| `pca_components_res3.pt` | `[20, 352]` | same for res3 |

Row 1 (PC2) is used as the fixed readout direction for stimulus synthesis.
