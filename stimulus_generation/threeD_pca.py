"""
3D PCA visualization of the synthesized optimization levels.

Paper reference: Results, Fig 2B-C.

The target layer's (spatially averaged) activations to every saved stimulus
are projected onto their first three principal components together with
the fixed linear classifier (Section 2.2.2). Levels form a graded sequence
from noisy pairs near the classifier boundary to the widely separated,
fully optimized pair.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.graph_objects as go

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA (GPU) is available')
else:
    device = torch.device('cpu')
    print('CUDA (GPU) is not available, falling back to CPU')


def perform_pca(x_optim_list, actv, classifier_weights=None, save_stim=False):
    """PCA (3 components) of the stored activations, plus the projection of
    the classifier weight vector onto the same components.

    Parameters
    ----------
    x_optim_list : list[torch.Tensor]
        Spatially averaged activations [nex, 32, 11] for each saved level
        (``save_stim=True``, the paper's setting).
    classifier_weights : torch.Tensor
        Weight of the target layer's fixed classifier [1, 352].

    Note: the ``save_stim=False`` branch expects global ``model`` and
    ``outputs`` objects and is not used in the paper.
    """
    all_data = torch.cat(x_optim_list, dim=0)
    print('all_data: ', all_data.shape)

    if save_stim == True:
        averaged_data = all_data
        print('averaged_data: ', averaged_data.shape)
    else:
        model(all_data)
        layer_data = outputs[actv]
        print('layer_data: ', layer_data.shape)
        averaged_data = torch.mean(layer_data, dim=(-2, -1))
        print('averaged_data: ', averaged_data.shape)

    # [N, channels, frames] -> [N, channels * frames]
    batch_size, num_channels, num_frames = averaged_data.size()
    reshaped_tensor = averaged_data.view(batch_size, num_channels * num_frames)
    print('reshaped_tensor: ', reshaped_tensor.shape)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_tensor.detach().cpu().numpy())

    pca_result_tensor = torch.tensor(pca_result, dtype=torch.float32).to(device)
    print('pca_result_tensor: ', pca_result_tensor.shape)
    print('pca.components_: ', pca.components_.shape)

    # Project the classifier weights onto the same 3 components
    classifier_projection = torch.mm(classifier_weights, torch.tensor(pca.components_.T, dtype=torch.float32).to(device))
    print('classifier_projection: ', classifier_projection.shape)

    return pca_result_tensor, classifier_projection


def plot_3d_scatter(nex_in, ax, pca1, pca2, pca3, label1, label2, color1, color2):
    """Scatter one level: first half of the batch (G1) and second half (G2)."""
    pca1_np = pca1.cpu().numpy()
    pca2_np = pca2.cpu().numpy()
    pca3_np = pca3.cpu().numpy()

    num_imgPerG = int(nex_in / 2)
    ax.scatter(pca1_np[:num_imgPerG], pca2_np[:num_imgPerG], pca3_np[:num_imgPerG], alpha=0.5, label=label1, c=color1)
    ax.scatter(pca1_np[num_imgPerG:], pca2_np[num_imgPerG:], pca3_np[num_imgPerG:], alpha=0.5, label=label2, c=color2)

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('3D Scatter plot of PCA1 vs PCA2 vs PCA3')
    ax.grid(True)
    ax.legend()


def plot_interactive_3d_scatter(nex_in, fig, pca1, pca2, pca3, label1, label2, color1, color2):
    """Plotly version of plot_3d_scatter."""
    pca1_np = pca1.cpu().numpy()
    pca2_np = pca2.cpu().numpy()
    pca3_np = pca3.cpu().numpy()

    num_imgPerG = int(nex_in / 2)

    fig.add_trace(go.Scatter3d(x=pca1_np[:num_imgPerG], y=pca2_np[:num_imgPerG], z=pca3_np[:num_imgPerG],
                               mode='lines+markers', marker=dict(color=color1),
                               name=label1))
    fig.add_trace(go.Scatter3d(x=pca1_np[num_imgPerG:], y=pca2_np[num_imgPerG:], z=pca3_np[num_imgPerG:],
                               mode='lines+markers', marker=dict(color=color2),
                               name=label2))


def generate_color(num_colors):
    """Blue and red palettes (dark -> light), one shade per level."""
    blue_palette = sns.color_palette("Blues", n_colors=num_colors)
    red_palette = sns.color_palette("Reds", n_colors=num_colors)

    blue_colors = [mcolors.to_hex(color) for color in blue_palette][::-1]
    red_colors = [mcolors.to_hex(color) for color in red_palette][::-1]

    return blue_colors, red_colors


def plot_3D_Visualization(pca_result_tensor, nex_in, classifier_projection, classifier_bias, save_img=False, base_folder=None, branch_folder=None, this_folder=None, display_classifier=False, classifier=None):
    """Static 3D PCA plot of all levels with the classifier plane
    (saved as ``3D_Visualization_videos.png``)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    blue_colors, red_colors = generate_color(int(pca_result_tensor.shape[0] / nex_in))
    cc = 0
    for i in range(0, pca_result_tensor.shape[0], nex_in):
        pca1 = pca_result_tensor[i:i + nex_in, 0]
        pca2 = pca_result_tensor[i:i + nex_in, 1]
        pca3 = pca_result_tensor[i:i + nex_in, 2]
        plot_3d_scatter(nex_in, ax, pca1, pca2, pca3, 'L' + str(cc) + '_G1', 'L' + str(cc) + '_G2', red_colors[cc], blue_colors[cc])
        cc = cc + 1

    # Classifier plane: w1*x + w2*y + w3*z + b = 0  ->  z = (-w1*x - w2*y - b) / w3
    if display_classifier == True:
        if classifier == 'CrossEntropyLoss':
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 10), torch.linspace(y_min, y_max, 10))
            zz_1 = (-classifier_projection[0, 0].detach().cpu() * xx - classifier_projection[0, 1].detach().cpu() * yy - classifier_bias[0].cpu()) / classifier_projection[0, 2].detach().cpu()
            zz_2 = (-classifier_projection[1, 0].detach().cpu() * xx - classifier_projection[1, 1].detach().cpu() * yy - classifier_bias[1].cpu()) / classifier_projection[1, 2].detach().cpu()

            xx_np = xx.detach().cpu().numpy()
            yy_np = yy.detach().cpu().numpy()
            zz_np_1 = zz_1.detach().cpu().numpy()
            zz_np_2 = zz_2.detach().cpu().numpy()

            ax.plot_surface(xx_np, yy_np, zz_np_1, color='red', alpha=0.3, label='Classifier Plane 1')
            ax.plot_surface(xx_np, yy_np, zz_np_2, color='blue', alpha=0.3, label='Classifier Plane 2')

        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 10), torch.linspace(y_min, y_max, 10))
            zz = (-classifier_projection[0, 0].detach().cpu() * xx - classifier_projection[0, 1].detach().cpu() * yy - classifier_bias.cpu()) / classifier_projection[0, 2].detach().cpu()
            xx_np = xx.detach().cpu().numpy()
            yy_np = yy.detach().cpu().numpy()
            zz_np = zz.detach().cpu().numpy()

            ax.plot_surface(xx_np, yy_np, zz_np, color='black', alpha=0.3, label='Classifier Plane')

    if save_img:
        save_path = os.path.join(base_folder, branch_folder, this_folder, '3D_Visualization_videos.png')
        plt.savefig(save_path)
    plt.show()


def plot_3D_Visualization_interactive(pca_result_tensor, nex_in, classifier_projection, classifier_bias, save_img=False, base_folder=None, branch_folder=None, this_folder=None, display_classifier=False, classifier=None):
    """Interactive (plotly) version, saved as ``3D_interactive_plot.html``."""
    fig = go.Figure()
    blue_colors, red_colors = generate_color(int(pca_result_tensor.shape[0] / nex_in))

    cc = 0
    for i in range(0, pca_result_tensor.shape[0], nex_in):
        pca1 = pca_result_tensor[i:i + nex_in, 0]
        pca2 = pca_result_tensor[i:i + nex_in, 1]
        pca3 = pca_result_tensor[i:i + nex_in, 2]
        plot_interactive_3d_scatter(nex_in, fig, pca1, pca2, pca3, 'L' + str(cc) + '_G1', 'L' + str(cc) + '_G2', red_colors[cc], blue_colors[cc])
        cc += 1

    if display_classifier:
        if classifier == 'CrossEntropyLoss':
            x_min, x_max = pca_result_tensor[:, 0].detach().cpu().numpy().min(), pca_result_tensor[:, 0].detach().cpu().numpy().max()
            y_min, y_max = pca_result_tensor[:, 1].detach().cpu().numpy().min(), pca_result_tensor[:, 1].detach().cpu().numpy().max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            zz_1 = (-classifier_projection[0, 0].detach().cpu() * xx - classifier_projection[0, 1].detach().cpu() * yy - classifier_bias[0].cpu()) / classifier_projection[0, 2].detach().cpu()
            zz_2 = (-classifier_projection[1, 0].detach().cpu() * xx - classifier_projection[1, 1].detach().cpu() * yy - classifier_bias[1].cpu()) / classifier_projection[1, 2].detach().cpu()
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz_1, opacity=0.3, name='Classifier Plane 1'))
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz_2, opacity=0.3, name='Classifier Plane 2'))
        else:
            x_min, x_max = pca_result_tensor[:, 0].detach().cpu().numpy().min(), pca_result_tensor[:, 0].detach().cpu().numpy().max()
            y_min, y_max = pca_result_tensor[:, 1].detach().cpu().numpy().min(), pca_result_tensor[:, 1].detach().cpu().numpy().max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            zz = (-classifier_projection[0, 0].detach().cpu() * xx - classifier_projection[0, 1].detach().cpu() * yy - classifier_bias.cpu()) / classifier_projection[0, 2].detach().cpu()
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.3, name='Classifier Plane'))

    fig.update_layout(scene=dict(xaxis_title='PCA1', yaxis_title='PCA2', zaxis_title='PCA3'),
                      title='3D Scatter plot of PCA1 vs PCA2 vs PCA3')
    if save_img:
        full_path = os.path.join(base_folder, branch_folder, this_folder, '3D_interactive_plot.html')
        full_path = os.path.normpath(full_path)
        fig.write_html(full_path)
    fig.show()


def calculate_angle_and_length_difference(vector1, vector2):
    """Angle (degrees) and absolute length difference between two vectors."""
    dot_product = torch.sum(vector1 * vector2)

    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)

    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    angle_radians = torch.acos(cosine_similarity)
    angle_degrees = angle_radians * 180 / 3.14159265359

    length_difference = abs(magnitude1 - magnitude2)

    return angle_degrees.item(), length_difference.item()
