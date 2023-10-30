import torch
from ezpred import metrics
from magnet import Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import BinaryConfusionMetric, metric
from torchmanager_core import view
from typing import Any
import numpy as np
import pandas as pd
import os

import data


def get_target_dict(num: int) -> dict[int, str]:

    if num == 1:
        dict_mod = {4:"DWIC"}
    elif num == 2:
        dict_mod = {3:"DWI"}
    elif num == 3:
        dict_mod = {3:"DWI", 4:"DWIC"}
    elif num == 4:
        dict_mod = {2:"FLAIR"}
    elif num == 5:
        dict_mod = {2:"FLAIR", 4:"DWIC"}
    elif num == 6:
        dict_mod = {2:"FLAIR", 3:"DWI"}
    elif num == 7:
        dict_mod = {2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 8:
        dict_mod = {1:"T2"}
    elif num == 9:
        dict_mod = {1:"T2", 4:"DWIC"}
    elif num == 10:
        dict_mod = {1:"T2", 3:"DWI"}
    elif num == 11:
        dict_mod = {1:"T2", 3:"DWI", 4:"DWIC"}
    elif num == 12:
        dict_mod = {1:"T2", 2:"FLAIR"}
    elif num == 13:
        dict_mod = {1:"T2", 2:"FLAIR", 4:"DWIC"}
    elif num == 14:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI"}
    elif num == 15:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 16:
        dict_mod = {0:"T1"}
    elif num == 17:
        dict_mod = {0:"T1", 4:"DWIC"}
    elif num == 18:
        dict_mod = {0:"T1", 3:"DWI"}
    elif num == 19:
        dict_mod = {0:"T1", 3:"DWI", 4:"DWIC"}
    elif num == 20:
        dict_mod = {0:"T1", 2:"FLAIR"}
    elif num == 21:
        dict_mod = {0:"T1", 2:"FLAIR", 4:"DWIC"}
    elif num == 22:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI"}
    elif num == 23:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 24:
        dict_mod = {0:"T1", 1:"T2"}
    elif num == 25:
        dict_mod = {0:"T1", 1:"T2", 4:"DWIC"}
    elif num == 26:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI"}
    elif num == 27:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI", 4:"DWIC"}
    elif num == 28:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR"}
    elif num == 29:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 4:"DWIC"}
    elif num == 30:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI"}
    elif num == 31:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    else:
        raise ValueError(f"num should be betwen 1 and 31, got {num}")

    return dict_mod

def test(cfg: TestingConfigs, /, target_dict: dict[int, str] = {0:'T1'}) -> Any:
    
    # load testing dataset
    testing_dataset = data.DatasetEZ_Node(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, node_num=str(cfg.node_num))
    
    # load checkpoint
    if cfg.model.endswith(".model"):
        manager = Manager.from_checkpoint(cfg.model, map_location=cfg.device)
        assert isinstance(manager, Manager), "Checkpoint is not a valid `ezpred.Manager`."
    else:
        raise NotImplementedError(f"Checkpoint {cfg.model} is currently not supported.")
    
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

    manager.target_dict = target_dict


    print(f'The best balanced accuracy on validation set occurs at {manager.current_epoch + 1} epoch number')

    # test checkpoint with validation cohort dataset (Last 10 patients)
    summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, empty_cache=False)
    # preds: list[torch.Tensor] = manager.predict(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    # print("Predictions: ", torch.cat([pred.argmax(-1) for pred in preds], -1).detach().cpu().numpy())

    # gt_vals: list[torch.Tensor] = [gt for _, gt in testing_dataset]
    # print("Ground-Truth: ", torch.cat([gt_val for gt_val in gt_vals], -1).detach().cpu().numpy())

    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)
    
    return summary['bal_accuracy'], manager.target_dict


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    ########################################################################################################
    ## evaluate for a single node and ALL 31 modality combinations

    # bal_accuracy = []

    # for i in range(1,32):
    #     dict_mod = get_target_dict(i)    
    #     bal_acc, mod_dict = test(configs, target_dict=dict_mod)
    #     bal_accuracy.append(bal_acc)
    #     print(f"Testing modality combination: {mod_dict}, accuracy is: {bal_acc}\n")

    # print(f"Final Testing modality combination mean is: {np.mean(bal_accuracy)}")


    # # dictionary of lists
    # predicted_acc_dict = {'Balanced Accuracy': bal_accuracy}    

    # df = pd.DataFrame(predicted_acc_dict)  

    # # saving the dataframe
    # path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/"  
    # save_path = os.path.join(path, "Node_948_Results")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # filename  = "Node_948_ALL_modality_combinations_trial5.xlsx"
    # save_filepath = os.path.join(save_path, filename)    

    # df.to_excel(save_filepath, index=False, sheet_name='Sheet1')

    ##########################################################################################################
    ## For ALL nodes but with a single modality combination {0:"T1", 1:"T2", 2:"FLAIR"} num == 28
       
    base_exp_model = configs.model

    dict_mod = get_target_dict(28)  

    num_trials = 5
    
    # Create empty lists to store results for each type (bal_accuracy)
    val_bal_acc_list = [[] for _ in range(num_trials)]

    # train
    for i in range(num_trials):
        print(f'\n\nStarting Trial {i+1} of Node number {configs.node_num} with Testing modality combination: {dict_mod}\n')

        configs.model = base_exp_model + "/exp_node" + str(configs.node_num) + "/magms_trial" + str(i+1) + ".exp/checkpoints/best_bal_accuracy.model"

        bal_acc, _ = test(configs, target_dict=dict_mod)

        val_bal_acc_list[i].append(bal_acc) 


    # Combine data
    row_data_val = [configs.node_num] + [val_bal_acc_list[j][0] for j in range(num_trials)]

    # Create a DataFrame
    headers_val = ['Node #', 'Val_Balanced_Accuracy_1', 'Val_Balanced_Accuracy_2', 'Val_Balanced_Accuracy_3', 'Val_Balanced_Accuracy_4', 'Val_Balanced_Accuracy_5']

    df_val = pd.DataFrame([row_data_val], columns=headers_val)

    # Saving to Excel
    # path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/"  
    path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Temporal_Lobe/"
    save_path = os.path.join(path, "Node_"+str(configs.node_num)+"_Results", "Eval_Results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename_val = "results_val.xlsx"
    save_filepath_val = os.path.join(save_path, filename_val)

    df_val.to_excel(save_filepath_val, index=False, sheet_name='Sheet1')

    print("\nDone!")

    
