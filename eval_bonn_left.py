# Evaluate the left-hemisphere nodes on the Bonn cohort (OpenNeuro ds004199)
#
# No training happens here. Each node reuses the three AddCohort checkpoints, which were trained on
# the original SMOTE-augmented training data with all five sequences and cross-sequence distillation.
# The Bonn cohort is therefore fully external: nothing about it influenced training or checkpoint
# selection.
#
# Those subjects have T1 and FLAIR only, so the sweep is the 3 non-empty subsets of {T1, FLAIR}. The
# absent sequences are dropped from the model's target_dict rather than being fed as all-zero
# branches - that substitution is the sequence-agnostic claim, so it has to be the real mechanism.
#
# Everything except the combination list and the output paths is shared with eval_left.py, and is
# imported from it rather than duplicated.
import os
import pandas as pd

import data
from ezpred.configs import TestingConfigs
from eval_left import get_target_dict, get_modality_name, load_manager, test


# The sequences this cohort actually has, as indices into get_target_dict's modality map.
T1, FLAIR = 0, 2


def get_t1_flair_combinations() -> list[int]:
    r"""
    The combination numbers of `get_target_dict` that use only T1 and/or FLAIR.

    Derived by filtering all 31 combinations rather than hard-coded, then checked to be exactly the
    3 non-empty subsets of {T1, FLAIR}, so a change to `get_target_dict` cannot silently mis-select.

    Returns: A `list` of the 3 combination numbers: 4 (FLAIR), 16 (T1) and 20 (T1-FLAIR).
    """
    present = {T1, FLAIR}

    combinations = [j for j in range(1, 32) if set(get_target_dict(j)[0]).issubset(present)]

    assert len(combinations) == 3, f"Expected 3 T1/FLAIR combinations, got {len(combinations)}."

    # every non-empty subset of {T1, FLAIR} must be covered exactly once
    covered = {frozenset(get_target_dict(j)[0]) for j in combinations}
    assert covered == {frozenset({T1}), frozenset({FLAIR}), frozenset({T1, FLAIR})}, \
        f"Combinations {combinations} do not cover the subsets of {{T1, FLAIR}}, got {covered}."

    return combinations


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    ############################################################################################################
    ## Score the three AddCohort trial checkpoints of each node on the 85 Bonn subjects, keeping every
    ## trial as its own column so the max/mean can be taken later.

    base_exp_model = configs.model  # e.g. ".../magmsforEZprediction/experiments"

    num_trials = 3

    # (mode, combination numbers, output filename). The same 85 subjects and the same features are
    # scored twice, differing only in which labels are used:
    #
    #   BONN            - the labels recovered from the BIDS source, 861 EZ over 455 nodes. This is
    #                     the balanced accuracy reported in the paper.
    #   BONN_ASEXPORTED - the labels exactly as /BonnData/ ships them, which are all zero. Every
    #                     node-subject pair counts as non-EZ, so BalancedAccuracyScore falls back to
    #                     specificity = sensitivity and the number is the non-EZ accuracy, i.e. the
    #                     true-negative rate for EZ detection.
    #
    # Running both in one pass shares the three checkpoint loads per node, which dominate the cost.
    GROUPS = [
        (data.EZMode.BONN, get_t1_flair_combinations(), "results_bonn_T1FLAIR3_trials.xlsx"),
        (data.EZMode.BONN_ASEXPORTED, get_t1_flair_combinations(), "results_bonn_asexported_T1FLAIR3_trials.xlsx"),
    ]

    path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/BonnCohort/Left_Hemis/"

    save_path = os.path.join(path, "Node_" + str(configs.node_num) + "_Results", "Eval_Results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # one row per modality combination, one column per trial, for each group
    bal_acc_per_trial: dict[str, dict[str, list[float]]] = {
        filename: {f"Trial_{i+1}": [] for i in range(num_trials)} for _, _, filename in GROUPS
    }

    # trials outermost so each checkpoint is loaded once and reused for every group and combination
    for i in range(num_trials):
        # the AddCohort checkpoints = WITH cross-sequence distillation, trained on all 5 sequences
        configs.model = base_exp_model + "/exp_node" + str(configs.node_num) + "/AddCohort" + "/magms_trial" + str(i+1) + ".exp/checkpoints/best_bal_accuracy.model"

        manager = load_manager(configs)

        for mode, combinations, filename_val in GROUPS:
            for j in combinations:
                dict_mod, list_mod = get_target_dict(j)

                print(f'\n\nStarting Trial {i+1} of Node number {configs.node_num} on {mode.value} with Testing modality combination: {dict_mod}\n')

                bal_acc, _, _, _ = test(configs, manager, target_dict=dict_mod, mode=mode)

                bal_acc_per_trial[filename_val][f"Trial_{i+1}"].append(bal_acc)

    for mode, combinations, filename_val in GROUPS:
        # rows are self-describing, so the per-node files stay readable once concatenated
        combination_names = [get_modality_name(get_target_dict(j)[0]) for j in combinations]

        df_val = pd.DataFrame({'Combination': combination_names, **bal_acc_per_trial[filename_val]})

        save_filepath_val = os.path.join(save_path, filename_val)

        df_val.to_excel(save_filepath_val, index=False, sheet_name='Sheet1')

        print(f"\nSaved {len(combination_names)} combinations x {num_trials} trials to {save_filepath_val}")

    print("\nDone!")
