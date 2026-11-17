"""Small shared helpers: layer activations and saving the run configuration."""

import os

import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')

from func_transforms import transform, clip


def get_actv(model, outputs, x_optim, actv, mask=None, transforms=None):
    """Return the activation of layer `actv` for stimulus `x_optim`.

    The stimulus goes through the same pipeline as during optimization
    (mask -> transforms -> clip to [-2, 2] -> DorsalNet). `outputs` is the
    dictionary filled by the forward hooks registered in the notebook.
    """
    x_optim_ = transform(x_optim * mask, transforms)
    x_optim_ = x_optim_.to(device)

    _ = model(clip(x_optim_)).to(device)

    layer_data = outputs[actv]

    return layer_data


def save_variables_to_file(obj, folder_path, filename="variables.txt"):
    """Write every attribute of a config object (e.g. Model_Variables) to a
    text file so each output folder documents the settings that produced it."""
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "w") as file:
        for attr, value in obj.__dict__.items():
            file.write(f"{attr} = {value}\n")
    print(f"Variables saved to {file_path}.")
