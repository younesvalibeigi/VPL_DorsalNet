"""
Linear readout classifier on top of the frozen DorsalNet.

Paper reference: Section 2.4.1 (Readout Architecture and Conditions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from func_transforms import transform

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')


class Classifier_centralpixel_binary(nn.Module):
    """Single linear layer + sigmoid reading the CENTRAL spatial unit of one
    or more DorsalNet layers.

    Reading only the central unit removes the spatial (height x width)
    dimensions while keeping all 32 channels and 11 frames, matching the
    readout used to define the stimulus-synthesis classifiers:

        single layer -> 32 x 11     = 352 features
        'All'        -> 32 x 11 x 4 = 1408 features

    DorsalNet's weights are frozen; the classifier weights (and bias) are the
    only trained parameters. They are initialized to zero so that early
    differences between readouts come from the layers' activations, not from
    the initial weights.
    """

    def __init__(self, base_model, classifier_w='default', dropout=None, spatial_size=164, readout='All'):
        """
        :param base_model: frozen DorsalNet model (forward hooks fill `outputs`)
        :param classifier_w: 'zero' (paper) initializes weights and bias to zero
        :param dropout: dropout probability on the features (None = no dropout)
        :param spatial_size: input frame size (kept for API compatibility)
        :param readout: 'All', 'singleLayer', 'modelOutput', 'res0', 'res1', 'res2' or 'res3'
        """
        super().__init__()
        self.base_model = base_model

        # Freeze DorsalNet
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout).to(device) if dropout else None

        # Feature block size of one layer: 32 channels x 11 frames
        self.block_size = 32 * 11
        if readout == 'All':
            self.classifier = nn.Linear(32 * 11 * 4, 1).to(device)
        else:
            self.classifier = nn.Linear(32 * 11, 1).to(device)

        if classifier_w == 'zero':
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x, mask, transforms, outputs, dropout=None, layer=None, ablate=False, readout=None):
        """
        Parameters
        ----------
        x : torch.Tensor [B, 3, 11, H, W]
        transforms : list or None
            Stochastic transforms (paper: random rescaling only).
        outputs : dict
            Filled by the forward hooks on res0-res3.
        layer : str
            Category of the current stimuli ('res0' or 'res2'); used by the
            stimulus-driven readout ('singleLayer').
        ablate : False, str or float
            Optional ablation (not used in the paper).
        readout : str
            Which layer(s) to read (see class docstring).

        Returns
        -------
        probability [B, 1], logits (numpy), weight (numpy), bias (numpy)
        """
        if transforms is not None:
            x = transform(x, transforms)

        model_output = self.base_model(x)

        # Select the layer(s) to read out
        if readout == 'All':
            L = torch.cat([outputs['res0'], outputs['res1'], outputs['res2'], outputs['res3']], dim=1)
        elif readout == 'singleLayer':
            # Stimulus-driven (adaptive-stage) readout: the layer each
            # stimulus was optimized for
            L = outputs[layer]
        elif readout == 'modelOutput':
            L = model_output
        elif readout == 'res0' or readout == 'res1' or readout == 'res2' or readout == 'res3':
            L = outputs[readout]

        # Central spatial unit: [B, C, T, H, W] -> [B, C * T]
        cz, cy, cx = L.shape[2] // 2, L.shape[3] // 2, L.shape[4] // 2
        central_value_flat = L[:, :, :, cy, cx].reshape((L.shape[0], -1))

        if dropout and self.dropout:
            central_value_flat = self.dropout(central_value_flat)

        # Optional ablation (not used in the paper; requires readout = 'All')
        if ablate is not False:
            weight = self.classifier.weight.clone()
            bias = self.classifier.bias
            s = self.block_size
            perm = torch.randperm(s, device=weight.device)
            # 'layer_tunedStimulus': ablate the layer the stimulus was made for
            if ablate == 'layer_tunedStimulus':
                ablate = layer
            if ablate == 'res0':
                central_value_flat[:, 0:s] = 0
            elif ablate == 'res1':
                central_value_flat[:, s:2 * s] = 0
            elif ablate == 'res2':
                central_value_flat[:, 2 * s:3 * s] = 0
            elif ablate == 'res3':
                weight[:, 3 * s:4 * s] = weight[:, 3 * s:4 * s][:, perm]
                central_value_flat[:, 3 * s:4 * s] = 0
            else:
                # `ablate` is a fraction in (0, 1): zero that proportion of
                # all readout features
                num_features = central_value_flat.size(1)
                k = int(ablate * num_features)

                idx = torch.randperm(num_features, device=central_value_flat.device)[:k]

                central_value_flat[:, idx] = 0

            logits = F.linear(central_value_flat, weight, bias)

        else:
            logits = self.classifier(central_value_flat)

        weight = self.classifier.weight.detach().cpu().numpy()
        bias = self.classifier.bias.detach().cpu().numpy()

        return self.sigmoid(logits), logits.detach().cpu().numpy(), weight, bias
