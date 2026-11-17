"""
Training and per-epoch evaluation of the ANN readout classifiers.

Paper reference: Sections 2.4.2 (Training Procedure) and 2.4.3 (Transfer
Testing).

For each stimulus category the classifier is trained on that category and,
after every training step, evaluated without weight updates on:

    train : the training stimuli themselves      (learning performance)
    loc   : the same stimuli at the shifted location (location transfer)
    stim  : the other category's stimuli          (stimulus-category transfer)

A fresh grayscale Gaussian noise patch (SD = var.guassian_noise_std, inside a
circle of radius var.crop_radius) is added on every training and evaluation
pass. Performance is later computed as 1 - |prediction - label|.
"""

import random

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from stimulus_dataset_utils import StimulusDatasetUtils


# -------------------------------------------------------
# Train + test runner for a single classifier
# -------------------------------------------------------
def _train_test_single_classifier(
        var, device, outputs,
        X_train, Y_train,
        X_loc_test, Y_loc_test,
        X_stim_test, Y_stim_test,
        classifier_model,
        SDU,
        track_weights=False,
        ablatedModel=False):
    """Train `classifier_model` for var.num_epochs steps and evaluate it on
    the train / location / stimulus sets after every step.

    Returns
    -------
    losses                    : BCE training loss per epoch
    losses_train              : |p - y| on the training set, per epoch
    losses_ablate             : |p - y| with ablation (= train when ablate is False)
    losses_test_loc           : |p - y| on the location-transfer set
    losses_test_stim          : |p - y| on the stimulus-transfer set
    confidences_*             : 2 * |p - 0.5| for the same sets
    classifier_weights_history: classifier weights after each step (if tracked)
    gradients_history         : weight gradients at each step (if tracked)
    classifier_logits         : logits on the training set, per epoch
    classifier_bias           : bias per epoch
    """
    # ------------------------------------
    # Output containers
    # ------------------------------------
    losses = []

    losses_train = []
    losses_ablate = []
    losses_test_loc = []
    losses_test_stim = []

    confidences_train = []
    confidences_ablate = []
    confidences_test_loc = []
    confidences_test_stim = []

    classifier_logits = []
    classifier_weights = []
    classifier_bias = []

    classifier_weights_history = [] if track_weights else None
    gradients_history = [] if track_weights else None

    # ------------------------------------
    # Loss + optimizer: BCE on the sigmoid output, Adam on the classifier
    # parameters (DorsalNet is frozen)
    # ------------------------------------
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(classifier_model.parameters(), lr=var.lr)

    if var.sch_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=var.step_size,
            gamma=var.gamma
        )

    # ------------------------------------
    # Training loop
    # ------------------------------------
    print(f"Start training {var.layer} with classifier: {classifier_model.__class__.__name__}")

    if var.threshold_performance is not None:
        if var.num_epochs_res0 is None:
            var.num_epochs = 10000  # large number; training stops when the threshold is reached
        else:
            var.num_epochs = var.num_epochs_res0

    for epoch in range(var.num_epochs):

        # ---------------------------------------------------
        # Add a fresh noise patch to the training stimuli
        # (moved back to CPU to reduce GPU memory)
        # ---------------------------------------------------
        X_aug = SDU.add_bw_gaussian_circle(
            X_train.to(device),
            mean=0.0,
            std=var.guassian_noise_std,
            radius=var.crop_radius
        ).cpu()

        if var.task == 'same_different':
            N_pairs = X_aug.size(0) // 2
            if var.differentCondWeight is None:
                batch_pairs = random.sample(range(N_pairs), var.batch_size)
            else:
                weights = [var.differentCondWeight if i % 4 in (0, 1) else 1 for i in range(N_pairs)]
                batch_pairs = random.choices(range(N_pairs), weights=weights, k=var.batch_size)

            X_batch = torch.cat([X_aug[2 * i:2 * i + 2] for i in batch_pairs], dim=0)
            Y_batch = torch.cat([Y_train[i].unsqueeze(1) for i in batch_pairs], dim=0)
        elif var.task == 'binary':
            # Random draw of `batch_size` stimuli from the training set
            N_pairs = X_aug.size(0)
            if var.differentCondWeight is None:
                batch_pairs = random.sample(range(N_pairs), var.batch_size)
            else:
                weights = [var.differentCondWeight if i % 2 in (0, 1) else 1 for i in range(N_pairs)]
                batch_pairs = random.choices(range(N_pairs), weights=weights, k=var.batch_size)

            X_batch = X_aug[batch_pairs]
            Y_batch = torch.cat([Y_train[i].unsqueeze(1) for i in batch_pairs], dim=0)

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # ---------------------------------------------------
        # Training step (weights updated after every step, as human
        # observers received feedback after every trial)
        # ---------------------------------------------------
        classifier_model.train()
        optimizer.zero_grad()

        # Optional microsaccade simulation (not used in the paper)
        if var.microsaccade:
            shift_px = random.randint(0, var.microsaccade_shift_px)
            position = random.choice(['bottom-right', 'top-right', 'bottom-left', 'top-left', 'top', 'bottom', 'left', 'right'])
            X_batch = SDU.move_center_square_to_corner2(X_batch, edge=var.crop_radius * 2,
                                                        gray_value=0.0, position=position, output_size=var.spatial_size,
                                                        shift_px=shift_px)

        out_train, logits_train, weight_train, bias_train = classifier_model(
            X_batch, None, var.transforms, outputs,
            dropout=var.dropout,
            layer=var.layer,
            ablate=False,  # ablation is only applied during testing
            readout=var.readout
        )

        loss = criterion(out_train, Y_batch)

        losses.append(loss.detach().cpu().numpy())

        loss.backward()

        # Save gradients (tracked for the 'All' readout)
        if track_weights:
            try:
                gradients_history.append(
                    classifier_model.classifier.weight.grad.detach().cpu().numpy()
                )
            except AttributeError:
                pass

        optimizer.step()

        if var.sch_lr:
            scheduler.step()

        # Save classifier weights (tracked for the 'All' readout)
        if track_weights:
            with torch.no_grad():
                try:
                    classifier_weights_history.append(
                        classifier_model.classifier.weight.detach().cpu().numpy()
                    )
                except AttributeError:
                    pass

        # ---------------------------------------------------
        # Evaluation after every step (no weight updates)
        # ---------------------------------------------------
        with torch.no_grad():
            classifier_model.eval()

            # ---------------- Training set ----------------
            X_ttrain = SDU.add_bw_gaussian_circle(
                X_train.to(device),
                mean=0.0,
                std=var.guassian_noise_std,
                radius=var.crop_radius
            )
            out_ttrain, logits_ttrain, weight_ttrain, bias_ttrain = classifier_model(
                X_ttrain, None, var.transforms, outputs, layer=var.layer, ablate=False, readout=var.readout
            )
            loss_ttrain = abs(out_ttrain - Y_train)
            conf_ttrain = ((out_ttrain - 0.5).abs() * 2)

            losses_train.append(loss_ttrain.detach().cpu().numpy())
            confidences_train.append(conf_ttrain.detach().cpu().numpy())
            classifier_logits.append(logits_ttrain)
            classifier_weights.append(weight_ttrain)
            classifier_bias.append(bias_ttrain)

            # ---------------- Ablation test ----------------
            # With var.ablate = False this is a second evaluation of the
            # training set. It is kept so the random-number stream (and
            # therefore every result) matches the original run.
            X_ablate = SDU.add_bw_gaussian_circle(
                X_train.to(device),
                mean=0.0,
                std=var.guassian_noise_std,
                radius=var.crop_radius
            )
            out_ablate, logits_ablate, weight_ablate, bias_ablate = classifier_model(
                X_ablate, None, var.transforms, outputs, layer=var.layer, ablate=var.ablate, readout=var.readout
            )
            loss_ablate = abs(out_ablate - Y_train)
            conf_ablate = ((out_ablate - 0.5).abs() * 2)

            losses_ablate.append(loss_ablate.detach().cpu().numpy())
            confidences_ablate.append(conf_ablate.detach().cpu().numpy())

            # ---------------- Location-transfer test ----------------
            X_loc = SDU.add_bw_gaussian_circle(
                X_loc_test.to(device),
                mean=0.0,
                std=var.guassian_noise_std,
                radius=var.crop_radius
            )
            out_loc, logits_loc, weight_loc, bias_loc = classifier_model(
                X_loc, None, var.transforms, outputs, layer=var.layer, ablate=False, readout=var.readout
            )
            loss_loc = abs(out_loc - Y_loc_test)
            conf_loc = ((out_loc - 0.5).abs() * 2)

            losses_test_loc.append(loss_loc.detach().cpu().numpy())
            confidences_test_loc.append(conf_loc.detach().cpu().numpy())

            # ---------------- Stimulus-category transfer test ----------------
            X_stim = SDU.add_bw_gaussian_circle(
                X_stim_test.to(device),
                mean=0.0,
                std=var.guassian_noise_std,
                radius=var.crop_radius
            )
            out_stim, logits_stim, weight_stim, bias_stim = classifier_model(
                X_stim, None, var.transforms, outputs, layer=var.layer, ablate=False, readout=var.readout
            )
            loss_stim = abs(out_stim - Y_stim_test)
            conf_stim = ((out_stim - 0.5).abs() * 2)

            losses_test_stim.append(loss_stim.detach().cpu().numpy())
            confidences_test_stim.append(conf_stim.detach().cpu().numpy())

        if epoch % var.printEpoch == 0:
            print(f"Epoch {epoch} | Train loss: {loss:.4f}")

        if var.threshold_performance is not None:
            if (1 - np.mean(loss_ttrain.detach().cpu().numpy())) > var.threshold_performance:
                var.num_epochs_res0 = epoch  # epoch at which the threshold was reached
                print(f"Threshold performance reached at epoch {epoch}. Stopping training.")
                return (
                    losses,
                    losses_train,
                    losses_ablate,
                    losses_test_loc,
                    losses_test_stim,
                    confidences_train,
                    confidences_ablate,
                    confidences_test_loc,
                    confidences_test_stim,
                    classifier_weights_history,
                    gradients_history,
                    classifier_logits,
                    classifier_bias
                )

    return (
        losses,
        losses_train,
        losses_ablate,
        losses_test_loc,
        losses_test_stim,
        confidences_train,
        confidences_ablate,
        confidences_test_loc,
        confidences_test_stim,
        classifier_weights_history,
        gradients_history,
        classifier_logits,
        classifier_bias
    )


# =====================================================================
# Train + test one readout architecture on both stimulus categories
# =====================================================================
def simultaneous_train_test_model_binary(var,
                                         device,
                                         outputs,
                                         xRes0_train,
                                         yRes0,
                                         xRes0_loc_test,
                                         xRes2_train,
                                         yRes2,
                                         xRes2_loc_test,
                                         classifier_singleLayerReadout_res0,
                                         classifier_singleLayerReadout_res2):
    """Run the binary task for the readout in ``var.readout``.

    * res0 classifier: trained on res0 stimuli; stimulus test uses res2 stimuli.
    * res2 classifier: trained on res2 stimuli; stimulus test uses res0 stimuli.

    Returns
    -------
    dict: results[<'res0'|'res2'>][var.readout][<key>] with keys
    losses, loss_train, loss_ablate, loss_loc, loss_stim, conf_train,
    conf_ablate, conf_loc, conf_stim, weights, gradients, logits, bias.
    """
    SDU = StimulusDatasetUtils(device=device)

    # ==========================================================
    # Classifier trained on res0 stimuli
    # ==========================================================
    var.layer = 'res0'
    (losses_res0,
     loss_train_res0,
     loss_ablate_res0,
     loss_test_loc_res0,
     loss_test_stim_res0,
     conf_train_res0,
     conf_ablate_res0,
     conf_test_loc_res0,
     conf_test_stim_res0,
     weights_res0,
     gradients_res0,
     classifier_logits_res0,
     classifier_bias_res0) = _train_test_single_classifier(
        var, device, outputs,
        xRes0_train, yRes0,
        xRes0_loc_test, yRes0,
        xRes2_train, yRes2,
        classifier_singleLayerReadout_res0,
        SDU,
        track_weights=(var.readout == 'All')
    )

    torch.cuda.empty_cache()

    # ==========================================================
    # Classifier trained on res2 stimuli
    # ==========================================================
    var.layer = 'res2'
    (losses_res2,
     loss_train_res2,
     loss_ablate_res2,
     loss_test_loc_res2,
     loss_test_stim_res2,
     conf_train_res2,
     conf_ablate_res2,
     conf_test_loc_res2,
     conf_test_stim_res2,
     weights_res2,
     gradients_res2,
     classifier_logits_res2,
     classifier_bias_res2) = _train_test_single_classifier(
        var, device, outputs,
        xRes2_train, yRes2,
        xRes2_loc_test, yRes2,
        xRes0_train, yRes0,
        classifier_singleLayerReadout_res2,
        SDU,
        track_weights=(var.readout == 'All')
    )

    torch.cuda.empty_cache()

    return {
        "res0": {
            var.readout: {
                "losses": losses_res0,
                "loss_train": loss_train_res0,
                "loss_ablate": loss_ablate_res0,
                "loss_loc": loss_test_loc_res0,
                "loss_stim": loss_test_stim_res0,
                "conf_train": conf_train_res0,
                "conf_ablate": conf_ablate_res0,
                "conf_loc": conf_test_loc_res0,
                "conf_stim": conf_test_stim_res0,
                "weights": weights_res0,
                "gradients": gradients_res0,
                "logits": classifier_logits_res0,
                "bias": classifier_bias_res0,
            }
        },
        "res2": {
            var.readout: {
                "losses": losses_res2,
                "loss_train": loss_train_res2,
                "loss_ablate": loss_ablate_res2,
                "loss_loc": loss_test_loc_res2,
                "loss_stim": loss_test_stim_res2,
                "conf_train": conf_train_res2,
                "conf_ablate": conf_ablate_res2,
                "conf_loc": conf_test_loc_res2,
                "conf_stim": conf_test_stim_res2,
                "weights": weights_res2,
                "gradients": gradients_res2,
                "logits": classifier_logits_res2,
                "bias": classifier_bias_res2
            }
        }
    }
