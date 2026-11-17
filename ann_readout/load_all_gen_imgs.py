"""Load the res0- and res2-targeted stimulus sets produced in Section 2."""

from load_raw_data import load_raw_data, load_raw_data_binary
from stimulus_dataset_utils import StimulusDatasetUtils


def load_all_gen_imgs(var, device='None'):
    """Return (beh_X_res0, beh_Y_res0, beh_X_res2, beh_Y_res2).

    Stimulus folders follow the naming used by the generation loop:
    ``<layer>_Inhb_<inhibitOtherLayers>_classifierPCA2__nex_2_var_<variation>``.
    """
    SDU = StimulusDatasetUtils(device=device)

    stim_address_res0 = var.main_address + '//' + 'res0' + '_Inhb_' + str(var.inhibitOtherLayers) + '_classifierPCA2__nex_2_var_' + var.variation_array[0]
    stim_address_res2 = var.main_address + '//' + 'res2' + '_Inhb_' + str(var.inhibitOtherLayers) + '_classifierPCA2__nex_2_var_' + var.variation_array[0]

    if var.task == 'same_different':
        if var.raw_data:
            beh_X_res0, beh_Y_res0 = load_raw_data(var, stim_address_res0, device=device)
            beh_X_res2, beh_Y_res2 = load_raw_data(var, stim_address_res2, device=device)
        else:
            beh_X_res0, beh_Y_res0 = SDU.recall_generated_data(var, stim_address_res0, device=device)
            beh_X_res2, beh_Y_res2 = SDU.recall_generated_data(var, stim_address_res2, device=device)
    elif var.task == 'binary':
        if var.raw_data:
            # Paper setting: raw optimized tensors
            beh_X_res0, beh_Y_res0 = load_raw_data_binary(var, stim_address_res0, device=device)
            beh_X_res2, beh_Y_res2 = load_raw_data_binary(var, stim_address_res2, device=device)
        else:
            beh_X_res0, beh_Y_res0 = SDU.recall_generated_data(var, stim_address_res0, task="binary", device=device)
            beh_X_res2, beh_Y_res2 = SDU.recall_generated_data(var, stim_address_res2, task="binary", device=device)

    return beh_X_res0, beh_Y_res0, beh_X_res2, beh_Y_res2
