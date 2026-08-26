import torch
from ezpred import metrics
from magnet import Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import BinaryConfusionMetric, metric
from torchmanager_core import view
from typing import Any
import pandas as pd
import os

import data


def get_target_dict(num: int) -> Any:

    if num == 1:
        dict_mod = {4:"DWIC"}
        list_mod = [0,0,0,0,1]
    elif num == 2:
        dict_mod = {3:"DWI"}
        list_mod = [0,0,0,1,0]
    elif num == 3:
        dict_mod = {3:"DWI", 4:"DWIC"}
        list_mod = [0,0,0,1,1]
    elif num == 4:
        dict_mod = {2:"FLAIR"}
        list_mod = [0,0,1,0,0]
    elif num == 5:
        dict_mod = {2:"FLAIR", 4:"DWIC"}
        list_mod = [0,0,1,0,1]
    elif num == 6:
        dict_mod = {2:"FLAIR", 3:"DWI"}
        list_mod = [0,0,1,1,0]
    elif num == 7:
        dict_mod = {2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [0,0,1,1,1]
    elif num == 8:
        dict_mod = {1:"T2"}
        list_mod = [0,1,0,0,0]
    elif num == 9:
        dict_mod = {1:"T2", 4:"DWIC"}
        list_mod = [0,1,0,0,1]
    elif num == 10:
        dict_mod = {1:"T2", 3:"DWI"}
        list_mod = [0,1,0,1,0]
    elif num == 11:
        dict_mod = {1:"T2", 3:"DWI", 4:"DWIC"}
        list_mod = [0,1,0,1,1]
    elif num == 12:
        dict_mod = {1:"T2", 2:"FLAIR"}
        list_mod = [0,1,1,0,0]
    elif num == 13:
        dict_mod = {1:"T2", 2:"FLAIR", 4:"DWIC"}
        list_mod = [0,1,1,0,1]
    elif num == 14:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI"}
        list_mod = [0,1,1,1,0]
    elif num == 15:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [0,1,1,1,1]
    elif num == 16:
        dict_mod = {0:"T1"}
        list_mod = [1,0,0,0,0]
    elif num == 17:
        dict_mod = {0:"T1", 4:"DWIC"}
        list_mod = [1,0,0,0,1]
    elif num == 18:
        dict_mod = {0:"T1", 3:"DWI"}
        list_mod = [1,0,0,1,0]
    elif num == 19:
        dict_mod = {0:"T1", 3:"DWI", 4:"DWIC"}
        list_mod = [1,0,0,1,1]
    elif num == 20:
        dict_mod = {0:"T1", 2:"FLAIR"}
        list_mod = [1,0,1,0,0]
    elif num == 21:
        dict_mod = {0:"T1", 2:"FLAIR", 4:"DWIC"}
        list_mod = [1,0,1,0,1]
    elif num == 22:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI"}
        list_mod = [1,0,1,1,0]
    elif num == 23:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [1,0,1,1,1]
    elif num == 24:
        dict_mod = {0:"T1", 1:"T2"}
        list_mod = [1,1,0,0,0]
    elif num == 25:
        dict_mod = {0:"T1", 1:"T2", 4:"DWIC"}
        list_mod = [1,1,0,0,1]
    elif num == 26:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI"}
        list_mod = [1,1,0,1,0]
    elif num == 27:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI", 4:"DWIC"}
        list_mod = [1,1,0,1,1]
    elif num == 28:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR"}
        list_mod = [1,1,1,0,0]
    elif num == 29:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 4:"DWIC"}
        list_mod = [1,1,1,0,1]
    elif num == 30:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI"}
        list_mod = [1,1,1,1,0]
    elif num == 31:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [1,1,1,1,1]
    else:
        raise ValueError(f"num should be betwen 1 and 31, got {num}")

    return dict_mod, list_mod


def get_modality_name(target_dict: dict[int, str]) -> str:
    r"""A readable key for a modality combination, e.g. `T1-FLAIR-DWIC`."""
    return "-".join(target_dict[k] for k in sorted(target_dict))


def get_dwic_free_combinations() -> list[int]:
    r"""
    The combination numbers of `get_target_dict` that do not contain DWIC.

    `get_target_dict` encodes `num` itself as a 5-bit mask whose least significant bit is DWIC
    (num 1 = 00001 = DWIC, num 16 = 10000 = T1), so the DWIC-free combinations are exactly the even
    numbers.

    Returns: A `list` of the 15 non-empty subsets of {T1, T2, FLAIR, DWI}.
    """
    combinations = [j for j in range(1, 32) if j % 2 == 0]
    assert len(combinations) == 15, f"Expected 15 DWIC-free combinations, got {len(combinations)}."
    for j in combinations:
        dict_mod, _ = get_target_dict(j)
        assert 4 not in dict_mod, f"Combination {j} ({dict_mod}) unexpectedly contains DWIC."
    return combinations


def load_manager(cfg: TestingConfigs, /) -> Manager:
    r"""
    Load a trained checkpoint once, so it can be reused across every modality combination instead of
    being reloaded for each of them.
    """
    if not cfg.model.endswith(".model"):
        raise NotImplementedError(f"Checkpoint {cfg.model} is currently not supported.")

    manager = Manager.from_checkpoint(cfg.model, map_location=cfg.device)
    assert isinstance(manager, Manager), "Checkpoint is not a valid `ezpred.Manager`."

    # set up confusion metrics
    bal_acc_fn = metrics.BalancedAccuracyScore()
    conf_met_fn = metrics.ConfusionMetrics(2)
    manager.metric_fns.update({
        "bal_accuracy": bal_acc_fn,
        "conf_met": conf_met_fn
        })

    for m in manager.metric_fns.values():
        if isinstance(m, BinaryConfusionMetric):
            m._class_index = 0 # since we consider non-EZ (class 0) as positive class

    print(f'The best balanced accuracy on validation set occurs at {manager.current_epoch + 1} epoch number')
    return manager


def test(cfg: TestingConfigs, manager: Manager, /, target_dict: dict[int, str] = {0:'T1'}, mode: data.EZMode = data.EZMode.ADD_17, return_probs: bool = False) -> Any:

    # load testing dataset
    testing_dataset = data.DatasetEZ_Node(cfg.batch_size, cfg.data_dir, mode=mode, node_num=str(cfg.node_num))

    conf_met_fn = manager.metric_fns["conf_met"]

    manager.target_dict = target_dict

    # test checkpoint with the requested validation cohort
    summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, empty_cache=False)

    gts = None
    probs_final_rounded = None

    # the per-patient probabilities need a second pass over the dataset, so only collect them when
    # they are actually going to be used
    if return_probs:
        preds: list[torch.Tensor] = manager.predict(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

        probs = torch.cat([pred.softmax(-1) for pred in preds], 0).detach().cpu().numpy()
        # print("Predictions: ", torch.cat([pred.argmax(-1) for pred in preds], -1).detach().cpu().numpy())
        # print(f"Probabilities: {probs}")

        gt_vals: list[torch.Tensor] = [gt for _, gt in testing_dataset]
        gts = torch.cat([gt_val for gt_val in gt_vals], -1).detach().cpu().numpy()
        # print("Ground-Truth: ", gts)

        probs_final = [probs[i][int(gt)] for i, gt in enumerate(gts)]
        # Round each value in the vector to 4 decimal places
        probs_final_rounded = [round(prob, 4) for prob in probs_final]
        # print("Final Probabilities: ", probs_final_rounded)

    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)

    return summary['bal_accuracy'], manager.target_dict, gts, probs_final_rounded


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    ############################################################################################################
    ## Evaluate one node of the left hemisphere on the cohort selected by --cohort.
    ##
    ##   --cohort add   (default) the additional validation cohort, scored with the AddCohort
    ##                  checkpoints. Two subject groups share the same 3 trial checkpoints:
    ##                    - the 17 subjects with all five sequences, on ALL 31 combinations
    ##                    - the 13 subjects without DWIC, on the 15 DWIC-free combinations
    ##                  DWIC is dropped from target_dict rather than supplied as an all-zero branch,
    ##                  which is what makes the model sequence-agnostic in the first place.
    ##                  Every trial is kept as its own column so the max or mean can be taken later.
    ##
    ##   --cohort orig  the original 10 held-out patients, scored with the Part_2 checkpoints on all
    ##                  31 combinations and averaged over the 3 trials. Written as one column of 31
    ##                  values named after the node, which is the layout
    ##                  combine_node_excel_sheet_results_eval_left.py concatenates.

    base_exp_model = configs.model  # e.g. ".../magmsforEZprediction/experiments"

    num_trials = 3

    # (checkpoint sub-directory, results root, [(mode, combination numbers, output filename), ...])
    if configs.cohort == "orig":
        experiment_dir = "Part_2"
        path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/"
        GROUPS = [
            (data.EZMode.TEST, list(range(1, 32)), "results_val_ALL_modalities_Part_2.xlsx"),
        ]
    else:
        experiment_dir = "AddCohort"
        path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/AddCohort/Left_Hemis/"
        GROUPS = [
            (data.EZMode.ADD_17, list(range(1, 32)), "results_add17_ALL31_trials.xlsx"),
            (data.EZMode.ADD_13, get_dwic_free_combinations(), "results_add13_DWICfree15_trials.xlsx"),
        ]

    save_path = os.path.join(path, "Node_" + str(configs.node_num) + "_Results", "Eval_Results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for mode, combinations, filename_val in GROUPS:
        # one row per modality combination, one column per trial
        bal_acc_per_trial: dict[str, list[float]] = {f"Trial_{i+1}": [] for i in range(num_trials)}

        for i in range(num_trials):
            configs.model = os.path.join(base_exp_model, f"exp_node{configs.node_num}", experiment_dir, f"magms_trial{i+1}.exp", "checkpoints", "best_bal_accuracy.model")

            # load the checkpoint once and reuse it for every modality combination
            manager = load_manager(configs)

            for j in combinations:
                dict_mod, list_mod = get_target_dict(j)

                print(f'\n\nStarting Trial {i+1} of Node number {configs.node_num} on {mode.value} with Testing modality combination: {dict_mod}\n')

                bal_acc, _, _, _ = test(configs, manager, target_dict=dict_mod, mode=mode)

                bal_acc_per_trial[f"Trial_{i+1}"].append(bal_acc)

        if configs.cohort == "orig":
            # historical layout: one column of 31 trial-averaged values, named after the node, so
            # that combine_node_excel_sheet_results_eval_left.py can concatenate the nodes
            mean_over_trials = [sum(trial[k] for trial in bal_acc_per_trial.values()) / num_trials for k in range(len(combinations))]

            df_val = pd.DataFrame(mean_over_trials, columns=["Node_" + str(configs.node_num)])
        else:
            # rows are self-describing, so the per-node files stay readable once concatenated
            combination_names = [get_modality_name(get_target_dict(j)[0]) for j in combinations]

            df_val = pd.DataFrame({'Combination': combination_names, **bal_acc_per_trial})

        save_filepath_val = os.path.join(save_path, filename_val)

        df_val.to_excel(save_filepath_val, index=False, sheet_name='Sheet1')

        print(f"\nSaved {len(combinations)} combinations x {num_trials} trials to {save_filepath_val}")

    print("\nDone!")
