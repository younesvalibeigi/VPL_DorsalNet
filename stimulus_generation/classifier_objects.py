"""
Fixed linear readout classifiers used to synthesize stimuli.

Paper reference: Section 2.2.2 (Linear Readout Classifiers).

For each residual layer a linear classifier with ONE output unit (sigmoid +
BCE) reads the activation at the central spatial unit of that layer
(32 channels x 11 frames = 352 features). Its weight vector is set to a
principal component of the layer's activations to the AirSim videos and
frozen: the classifier is never trained, only the stimulus is optimized
against it.
"""

import torch
from torch import nn

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')


def classifier_objects(x_shape,
                       weights='rand',
                       pca_num=None,
                       act='res1',
                       pca_components=None,
                       random_seed=None,
                       requireGrad=False,
                       loss_func='CrossEntropyLoss',
                       crossEntFeat=2,
                       reduction='mean',
                       central_pixel=True):
    """Build a fixed linear classifier for layer `act`.

    Parameters
    ----------
    x_shape : int
        Number of stimuli in the batch (2 = one pair). The first half gets
        label 0, the second half label 1.
    weights : str
        'pca' (paper), 'rand', 'ones', 'half', 'pca1' or 'pca2'.
    pca_num : int
        Which principal component to use when weights == 'pca' (paper: 2).
    pca_components : torch.Tensor
        [n_components, 352] principal components for this layer.
    requireGrad : bool
        False (paper) freezes weights and bias.
    loss_func : str
        'BCELoss' (paper, 1 output unit) or 'CrossEntropyLoss'.
    central_pixel : bool
        True (paper): input is the central spatial unit (32 x 11 = 352 features).

    Returns
    -------
    linclassifier, CEloss, linlabels, bias_values
    """
    if loss_func == 'CrossEntropyLoss':
        CEloss = nn.CrossEntropyLoss()
        out_feat = crossEntFeat
    elif loss_func == 'BCELoss':
        CEloss = nn.BCELoss()
        if reduction == 'none':
            CEloss = nn.BCELoss(reduction='none')
        else:
            CEloss = nn.BCELoss()
        out_feat = 1

    # in_features must correspond to the activation layer
    if (act == 'res0' or act == 'res1' or act == 'res2' or act == 'res3'):
        if central_pixel:
            in_feat = 352                   # 32 channels x 11 frames
        else:
            in_feat = 32 * 11 * 45 * 45
    elif (act == 's1'):
        if central_pixel:
            in_feat = 704
        else:
            in_feat = 64 * 11 * 48 * 48     # may need checking for s1

    # The bias keeps PyTorch's default initialization and is frozen below.
    linclassifier = torch.nn.Linear(in_features=(in_feat), out_features=out_feat, bias=True)
    for param in linclassifier.parameters():
        param.requires_grad = requireGrad  # False: freeze all weights

    if requireGrad == False:
        if weights == 'ones':
            with torch.no_grad():
                linclassifier.weight = nn.Parameter(torch.ones_like(linclassifier.weight))
        elif weights == 'rand':
            with torch.no_grad():
                linclassifier.weight = nn.Parameter(torch.randn_like(linclassifier.weight))
        elif weights == 'half':
            with torch.no_grad():
                # Half of the weights 0, the other half 1
                num_features = linclassifier.weight.size(1)
                half_features = num_features // 2
                linclassifier.weight[:, :half_features] = 0
                linclassifier.weight[:, half_features:] = 1
        elif weights == 'pca1' and pca_components is not None:
            with torch.no_grad():
                if loss_func == 'CrossEntropyLoss':
                    linclassifier.weight[0] = pca_components[0, :].to(device)
                    linclassifier.weight[1] = pca_components[0, :].to(device)
                elif loss_func == 'BCELoss':
                    linclassifier.weight[0] = pca_components[0, :].to(device)
        elif weights == 'pca2' and pca_components is not None:
            with torch.no_grad():
                if loss_func == 'CrossEntropyLoss':
                    linclassifier.weight[0] = pca_components[1, :].to(device)
                    linclassifier.weight[1] = pca_components[1, :].to(device)
                elif loss_func == 'BCELoss':
                    linclassifier.weight[0] = pca_components[1, :].to(device)
        elif weights == 'pca' and pca_components is not None:
            # Paper setting: weight vector = principal component `pca_num`
            with torch.no_grad():
                if loss_func == 'CrossEntropyLoss':
                    raise Exception("CrossEntropyLoss does not work with PCA")
                elif loss_func == 'BCELoss':
                    print('pca_num:', pca_num)
                    linclassifier.weight[0] = pca_components[pca_num - 1, :].to(device)

    # Category labels: first half of the batch -> 0, second half -> 1
    linlabels = torch.as_tensor([0] * int(x_shape / 2) + [1] * int(x_shape / 2))
    bias_values = linclassifier.bias.data

    return linclassifier, CEloss, linlabels, bias_values
