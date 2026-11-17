"""Loss plots saved alongside each synthesized stimulus set (paper Fig 1F-G)."""

import os

import numpy as np
import matplotlib.pyplot as plt


def make_directory_path(name):
    if not os.path.exists(name):
        os.makedirs(name)


def plot_losses(losses, thisfolder, group=None, level=None):
    """Total loss and individual terms over iterations."""
    labels = ['distance', 'mean penalty', 'colour penalty', 'contrast penalty', 'mixed contrast penalty']
    plt.plot(np.arange(losses.shape[0]), np.sum(losses, axis=1), '.k', label='total loss')
    for j in np.arange(losses.shape[1]):
        if (j == 0):
            alpha = 1
        else:
            alpha = 0.3
        plt.plot(np.arange(losses.shape[0]), losses[:, j], '.', label=labels[j], alpha=alpha)
    if (level != None and group != None):
        plt.savefig(os.path.join(thisfolder, 'losses_G' + str(group) + '_L' + str(level) + '.png'))
    else:
        plt.savefig(os.path.join(thisfolder, 'losses.png'))
    plt.show()


def plot_losses_layers(losses, thisfolder, group=None, level=None):
    """Classifier term of each residual layer over iterations.

    During synthesis only the target layer's term is optimized; this plot
    shows that the classification loss falls further at the target layer
    than at the non-target layers.
    """
    labels = ['loss-res0', 'loss-res1', 'loss-res2', 'loss-res3']

    for i in range(losses.shape[1]):
        plt.plot(range(losses.shape[0]), losses[:, i], label=labels[i], linewidth=1)

    plt.xlabel('Iterations')
    plt.ylabel('Loss Values')
    plt.legend()

    if (level != None and group != None):
        plt.savefig(os.path.join(thisfolder, 'losses_G' + str(group) + '_L' + str(level) + '.png'))
    else:
        plt.savefig(os.path.join(thisfolder, 'losses.png'))
    plt.show()


def plot_losses_startFullNoise(losses, thisfolder, group=None, level=None):
    """Distance loss of the graded series (Section 2.2.5) over iterations."""
    labels = ['loss-activation layer distance between two stimuli']

    for i in range(losses.shape[1]):
        plt.plot(range(losses.shape[0]), losses[:, i], label=labels[i], linewidth=1)

    plt.xlabel('Iterations')
    plt.ylabel('Loss Values')
    plt.legend()

    if (level != None and group != None):
        plt.savefig(os.path.join(thisfolder, 'losses_G' + str(group) + '_L' + str(level) + '.png'))
    else:
        plt.savefig(os.path.join(thisfolder, 'losses.png'))
    plt.show()
