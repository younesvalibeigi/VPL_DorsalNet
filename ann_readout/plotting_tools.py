"""
Summary figures for one readout architecture of the ANN experiment.

Performance is defined as 1 - |prediction - label| (Section 2.5.2). The
specificity index plotted here is

    SI = (P_train - P_test) / (P_train - P_train[epoch 0]),  clipped to [-1, 1]

i.e. the drop in performance on the transfer set, relative to the
improvement achieved on the training set.
"""

import numpy as np
import matplotlib.pyplot as plt


class LossPlotter:
    """Plots for the results dictionary returned by
    ``simultaneous_train_test_model_binary``:
    ``results[<'res0'|'res2'>][<readout>][<key>]``.
    """

    def __init__(self, results):
        self.results = results

        # Global style
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update({
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.weight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "legend.fontsize": 10,
        })

    # ---------------------- Helper methods ----------------------
    @staticmethod
    def avg_levels(arr, indices):
        arr = arr.squeeze(-1)
        return arr[:, indices].mean(axis=1)

    @staticmethod
    def get_same_indices(n_levels):
        return sorted(list(range(2, n_levels, 4)) + list(range(3, n_levels, 4)))

    @staticmethod
    def get_diff_indices(n_levels):
        return sorted(list(range(0, n_levels, 4)) + list(range(1, n_levels, 4)))

    @staticmethod
    def avg_all(arr):
        """[epochs, n_stimuli, 1] -> mean over stimuli -> [epochs]"""
        arr = arr.squeeze(-1)
        return arr.mean(axis=1)

    @staticmethod
    def avg_group(values, groups):
        return np.array([np.mean(values[list(g)]) for g in groups])

    # -----------------------------------------------------------------
    # Performance over epochs + last-epoch performance per level group
    # -----------------------------------------------------------------
    def plot_binary_summary_4panel(self, clf_key="singleLayer", epoch=-1):
        """2 x 2 figure:
        top-left    : location test, performance over epochs (train vs loc-test)
        top-right   : stimulus test, performance over epochs (train vs stim-test)
        bottom-left : location test at `epoch`, per level group
        bottom-right: stimulus test at `epoch`, per level group
        """
        results = self.results

        res0_train = np.array(results["res0"][clf_key]["loss_train"])
        res2_train = np.array(results["res2"][clf_key]["loss_train"])

        res0_loc = np.array(results["res0"][clf_key]["loss_loc"])
        res2_loc = np.array(results["res2"][clf_key]["loss_loc"])

        res0_stim = np.array(results["res0"][clf_key]["loss_stim"])
        res2_stim = np.array(results["res2"][clf_key]["loss_stim"])

        n_levels = res0_train.shape[1]

        # Group the stimuli of each level (2 per level in the binary task)
        all_groups = [(i, i + 1) for i in range(0, n_levels, 2)]

        def avg_group(values_1d, groups):
            return np.array([np.mean(values_1d[list(g)]) for g in groups])

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f"Binary Summary — {clf_key}", fontsize=18, fontweight="bold")
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        # Location test over epochs
        ax = axes[0, 0]
        ax.plot(1 - self.avg_all(res0_train), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - self.avg_all(res2_train), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - self.avg_all(res0_loc), "-", color="blue", label="Res0 Loc-Test")
        ax.plot(1 - self.avg_all(res2_loc), "-", color="green", label="Res2 Loc-Test")
        ax.set_title("LOC — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        # Stimulus test over epochs
        ax = axes[0, 1]
        ax.plot(1 - self.avg_all(res0_train), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - self.avg_all(res2_train), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - self.avg_all(res0_stim), "-", color="blue", label="Res0 Stim-Test")
        ax.plot(1 - self.avg_all(res2_stim), "-", color="green", label="Res2 Stim-Test")
        ax.set_title("STIM — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        # Location test at `epoch`, per level group
        ax = axes[1, 0]
        r0_tr_last = res0_train[epoch, :]
        r2_tr_last = res2_train[epoch, :]
        r0_loc_last = res0_loc[epoch, :]
        r2_loc_last = res2_loc[epoch, :]

        ax.plot(1 - avg_group(r0_tr_last, all_groups), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - avg_group(r2_tr_last, all_groups), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - avg_group(r0_loc_last, all_groups), "-", color="blue", label="Res0 Loc-Test")
        ax.plot(1 - avg_group(r2_loc_last, all_groups), "-", color="green", label="Res2 Loc-Test")
        ax.set_title(f"LOC — Last epoch (epoch={epoch}) over level-groups", fontweight="bold")
        ax.set_xlabel("Level group index")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        # Stimulus test at `epoch`, per level group
        ax = axes[1, 1]
        r0_stim_last = res0_stim[epoch, :]
        r2_stim_last = res2_stim[epoch, :]

        ax.plot(1 - avg_group(r0_tr_last, all_groups), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - avg_group(r2_tr_last, all_groups), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - avg_group(r0_stim_last, all_groups), "-", color="blue", label="Res0 Stim-Test")
        ax.plot(1 - avg_group(r2_stim_last, all_groups), "-", color="green", label="Res2 Stim-Test")
        ax.set_title(f"STIM — Last epoch (epoch={epoch}) over level-groups", fontweight="bold")
        ax.set_xlabel("Level group index")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        plt.show()

    # -----------------------------------------------------------------
    # Performance and confidence over epochs
    # -----------------------------------------------------------------
    def plot_binary_summary_performance_confidence(self, clf_key="singleLayer", epoch=-1):
        """2 x 2 figure:
        top    : performance over epochs (location test, stimulus test)
        bottom : confidence 2|p - 0.5| over epochs (location test, stimulus test)
        """
        results = self.results

        res0_train = np.array(results["res0"][clf_key]["loss_train"])
        res2_train = np.array(results["res2"][clf_key]["loss_train"])

        res0_loc = np.array(results["res0"][clf_key]["loss_loc"])
        res2_loc = np.array(results["res2"][clf_key]["loss_loc"])

        res0_stim = np.array(results["res0"][clf_key]["loss_stim"])
        res2_stim = np.array(results["res2"][clf_key]["loss_stim"])

        res0_train_conf = np.array(results["res0"][clf_key]["conf_train"])
        res2_train_conf = np.array(results["res2"][clf_key]["conf_train"])

        res0_loc_conf = np.array(results["res0"][clf_key]["conf_loc"])
        res2_loc_conf = np.array(results["res2"][clf_key]["conf_loc"])

        res0_stim_conf = np.array(results["res0"][clf_key]["conf_stim"])
        res2_stim_conf = np.array(results["res2"][clf_key]["conf_stim"])

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f"Binary Summary — {clf_key}", fontsize=18, fontweight="bold")
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        # Location test: performance
        ax = axes[0, 0]
        ax.plot(1 - self.avg_all(res0_train), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - self.avg_all(res2_train), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - self.avg_all(res0_loc), "-", color="blue", label="Res0 Loc-Test")
        ax.plot(1 - self.avg_all(res2_loc), "-", color="green", label="Res2 Loc-Test")
        ax.set_title("LOC — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        # Stimulus test: performance
        ax = axes[0, 1]
        ax.plot(1 - self.avg_all(res0_train), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(1 - self.avg_all(res2_train), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(1 - self.avg_all(res0_stim), "-", color="blue", label="Res0 Stim-Test")
        ax.plot(1 - self.avg_all(res2_stim), "-", color="green", label="Res2 Stim-Test")
        ax.set_title("STIM — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance (1 – Loss)")
        ax.grid(True)
        ax.legend()

        # Location test: confidence
        ax = axes[1, 0]
        ax.plot(self.avg_all(res0_train_conf), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(self.avg_all(res2_train_conf), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(self.avg_all(res0_loc_conf), "-", color="blue", label="Res0 Loc-Test")
        ax.plot(self.avg_all(res2_loc_conf), "-", color="green", label="Res2 Loc-Test")
        ax.set_title("LOC — Confidence over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Confidence")
        ax.grid(True)
        ax.legend()

        # Stimulus test: confidence
        ax = axes[1, 1]
        ax.plot(self.avg_all(res0_train_conf), "-", color="#66b3ff", label="Res0 Train")
        ax.plot(self.avg_all(res2_train_conf), "-", color="#8cd98c", label="Res2 Train")
        ax.plot(self.avg_all(res0_stim_conf), "-", color="blue", label="Res0 Stim-Test")
        ax.plot(self.avg_all(res2_stim_conf), "-", color="green", label="Res2 Stim-Test")
        ax.set_title("STIM — Confidence over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Confidence")
        ax.grid(True)
        ax.legend()

        plt.show()

    # -----------------------------------------------------------------
    # Ablation (not used in the paper)
    # -----------------------------------------------------------------
    def plot_binary_ablation(self, clf_key="singleLayer", ablation='All'):
        """Training vs ablated performance and the resulting specificity index."""
        results = self.results

        res0_train = 1 - self.avg_all(np.array(results["res0"][clf_key]["loss_train"]))
        res2_train = 1 - self.avg_all(np.array(results["res2"][clf_key]["loss_train"]))

        res0_ablate = 1 - self.avg_all(np.array(results["res0"][clf_key]["loss_ablate"]))
        res2_ablate = 1 - self.avg_all(np.array(results["res2"][clf_key]["loss_ablate"]))

        si_res0_ablate = np.clip((res0_train - res0_ablate) / (res0_train - res0_train[0] + 1e-8), -1, 1.0)
        si_res2_ablate = np.clip((res2_train - res2_ablate) / (res2_train - res2_train[0] + 1e-8), -1, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(f"Binary: Ablation — {ablation}", fontsize=18, fontweight="bold")
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        ax = axes[0]
        ax.plot(res0_train, "-", color="#66b3ff", label="Res0 Train")
        ax.plot(res0_ablate, "-", color="blue", label="Res0 Ablate")
        ax.set_title("Ablation res0 stim — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance")
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(res2_train, "-", color="#8cd98c", label="Res2 Train")
        ax.plot(res2_ablate, "-", color="green", label="Res2 Ablate")
        ax.set_title("Ablation res2 stim — Performance over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance")
        ax.grid(True)
        ax.legend()

        ax = axes[2]
        ax.plot(si_res0_ablate, "-", color="blue", label="Res0 SI")
        ax.plot(si_res2_ablate, "-", color="green", label="Res2 SI")
        ax.set_title("Ablation Specificity Index", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Specificity Index")
        ax.set_ylim([-1, 1])
        ax.grid(True)
        ax.legend()

        plt.show()

    # -----------------------------------------------------------------
    # Specificity index over epochs
    # -----------------------------------------------------------------
    def plot_binary_specificity(self, clf_key="singleLayer", epoch=-1):
        """Location- and stimulus-transfer specificity index over epochs.

            SI = (P_train - P_test) / (P_train - P_train[epoch 0]), clipped to [-1, 1]

        SI near 0: full transfer; SI near 1: learning is specific to the
        trained location / stimulus.
        """
        results = self.results

        res0_train = 1 - self.avg_all(np.array(results["res0"][clf_key]["loss_train"]))
        res2_train = 1 - self.avg_all(np.array(results["res2"][clf_key]["loss_train"]))

        res0_loc = 1 - self.avg_all(np.array(results["res0"][clf_key]["loss_loc"]))
        res2_loc = 1 - self.avg_all(np.array(results["res2"][clf_key]["loss_loc"]))
        res0_stim = 1 - self.avg_all(np.array(results["res0"][clf_key]["loss_stim"]))
        res2_stim = 1 - self.avg_all(np.array(results["res2"][clf_key]["loss_stim"]))

        si_res0_loc = np.clip((res0_train - res0_loc) / (res0_train - res0_train[0] + 1e-8), -1, 1.0)
        si_res2_loc = np.clip((res2_train - res2_loc) / (res2_train - res2_train[0] + 1e-8), -1, 1.0)

        si_res0_stim = np.clip((res0_train - res0_stim) / (res0_train - res0_train[0] + 1e-8), -1, 1.0)
        si_res2_stim = np.clip((res2_train - res2_stim) / (res2_train - res2_train[0] + 1e-8), -1, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Binary: Specificity Summary — {clf_key}", fontsize=18, fontweight="bold")
        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        ax = axes[0]
        ax.plot(si_res0_loc, "-", color="blue", label="Res0 Loc-Test")
        ax.plot(si_res2_loc, "-", color="green", label="Res2 Loc-Test")
        ax.set_title("LOC — Specificity over epoch", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Specificity Index")
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(si_res0_stim, "-", color="blue", label="Res0 Stim-Test")
        ax.plot(si_res2_stim, "-", color="green", label="Res2 Stim-Test")
        ax.set_title("STIM — Specificity over epochs", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Specificity")
        ax.grid(True)
        ax.legend()

        for ax in axes:
            ax.set_ylim(0.5, 1)
        plt.show()

    # -----------------------------------------------------------------
    # BCE training loss
    # -----------------------------------------------------------------
    def plot_binary_BCELoss(self, clf_key="singleLayer", epoch=-1):
        results = self.results

        res0_losses = np.array(results["res0"][clf_key]["losses"])
        res2_losses = np.array(results["res2"][clf_key]["losses"])

        plt.figure(figsize=(8, 6))

        plt.plot(res0_losses, "-", color="blue", label="Res0 BCE Loss")
        plt.plot(res2_losses, "-", color="green", label="Res2 BCE Loss")

        plt.title("BCE Loss over epochs", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.grid(True)
        plt.legend()

        plt.show()

    # -----------------------------------------------------------------
    # Classifier weights per layer ('All' readout only)
    # -----------------------------------------------------------------
    def plot_batchwise_weights2(self,
                                params="weights",
                                readout=None,
                                histogram=False,
                                epoch=-1,
                                res_layers=[1, 2, 3, 4]):
        """Mean |weight| (or |gradient|) of each layer's block of the 'All'
        readout over epochs, or their histogram at `epoch`.
        Use ``res_layers`` with indices 0-3 (res0-res3)."""
        results = self.results

        res_layers = np.array(res_layers)

        layer_colors = ["blue", "orange", "green", "red"]
        layer_names = ["Res0", "Res1", "Res2", "Res3"]

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True, sharey=True)
        if histogram:
            fig.suptitle("Distribution of Batchwise Classifier " + str(params) + " at Epoch " + str(epoch),
                         fontsize=22, fontweight="bold")
        else:
            fig.suptitle("Batchwise Classifier " + str(params) + " Over Epochs\n(Averaged per Layer)",
                         fontsize=22, fontweight="bold")
        plt.subplots_adjust(wspace=0.25)

        def extract_weights(entry, params="weights"):
            W = np.array(entry[params])
            W = np.squeeze(W, axis=tuple(i for i in range(3, W.ndim)))
            if W.ndim != 3 or W.shape[1] != 1:
                raise ValueError(f"Unexpected weight shape after squeezing: {W.shape}")
            return W[:, 0, :]  # (epochs, C)

        for ax, stim_key, title in [(axes[0], "res0", "Res0 Stimulus "), (axes[1], "res2", "Res2 Stimulus ")]:
            if readout is not None:
                W = extract_weights(results[stim_key][readout], params=params)
            else:
                W = extract_weights(results[stim_key]["batchwise"], params=params)

            layer_size = W.shape[1] // 4

            W_blocks = np.stack([
                abs(W[:, 0:layer_size]).mean(axis=1),
                abs(W[:, layer_size:int(layer_size * 2)]).mean(axis=1),
                abs(W[:, int(layer_size * 2):int(layer_size * 3)]).mean(axis=1),
                abs(W[:, int(layer_size * 3):int(layer_size * 4)]).mean(axis=1),
            ], axis=1)

            W_epochs = abs(W[epoch, :])
            if histogram:
                for i in reversed(res_layers):
                    ax.hist(W_epochs[i * layer_size:(i + 1) * layer_size],
                            bins=100, alpha=0.6, color=layer_colors[i], label=layer_names[i])
            else:
                for i in res_layers:
                    ax.plot(W_blocks[:, i], color=layer_colors[i], label=layer_names[i])

            ax.set_title(title + params, fontweight="bold")
            ax.set_xlabel("Epoch", fontweight="bold")
            ax.set_ylabel("Average " + params, fontweight="bold")
            ax.grid(True)
            ax.legend()

        plt.show()
