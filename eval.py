import torch
from ezpred import metrics
from magnet import Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import metric
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
    # load whole brain dataset
    # validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, fold_no=cfg.fold_no)

    # load whole brain original validation dataset
    # validation_dataset = data.DatasetEZ_WB_Val_Original(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, fold_no=cfg.fold_no)

    # validation_dataset = data.DatasetEZ_WB_ALL_Original(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, fold_no=cfg.fold_no)

    validation_dataset = data.DatasetEZ_WB_SubGroupAnalysis(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, fold_no=cfg.fold_no)

    # load whole brain control dataset
    # validation_dataset = data.DatasetEZ_WB_Control(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE)
    
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
        "val_bal_accuracy": bal_acc_fn,
        "conf_met": conf_met_fn
        })

    ## 0:T1, 1:T2, 2:FLAIR, 3:DWI, 4:DWIC
    # manager.target_dict = {
    #     0: "T1",
    #     1: "T2",
    #     2: "FLAIR",
    #     3: "DWI",
    #     4: "DWIC",
    # }

    manager.target_dict = target_dict

    # print(manager.target_dict)

    # print(f'The best accuracy on validation set occurs at {manager.current_epoch + 1} epoch number')

    # test checkpoint with validation dataset
    summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    # preds = manager.predict(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)
    
    # test checkpoint with testing dataset
    # summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    # if conf_met_fn.results is not None:
    #     summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    # return summary['accuracy'], manager.target_dict, preds # only if predictions are needed
    return summary['accuracy'], manager.target_dict


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    # dict_mod = get_target_dict(31)    
    # # acc, mod_dict, preds = test(configs, target_dict=dict_mod)
    # acc, mod_dict = test(configs, target_dict=dict_mod)
    
    # print(f"Testing modality combination: {mod_dict}, accuracy is: {acc}\n")

    # predicted_acc = []

    # for i in range(1654):
    #     preds_1 = preds[i].squeeze(0)
    #     preds_f = torch.argmax(preds_1, dim=0)
    #     predicted_acc.append(preds_f.item())
    
    # # dictionary of lists
    # predicted_acc_dict = {'Accuracy': predicted_acc}    

    # df = pd.DataFrame(predicted_acc_dict)  

    # # saving the dataframe
    # path = "/home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/"  
    # save_path = os.path.join(path, "Control_Data_Results")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # filename  = "control_data_accuracy_model_fold5.csv"
    # save_filepath = os.path.join(save_path, filename)

    # df.to_csv(save_filepath, header=False, index=False)


    accuracy = []

    for i in range(1,32):
        dict_mod = get_target_dict(i)    
        acc, mod_dict = test(configs, target_dict=dict_mod)
        accuracy.append(acc)
        print(f"Testing modality combination: {mod_dict}, accuracy is: {acc}\n")

    print(f"Final Testing modality combination mean is: {np.mean(accuracy)}")

    # dictionary of lists
    predicted_acc_dict = {'Accuracy': accuracy}    

    df = pd.DataFrame(predicted_acc_dict)  

    # saving the dataframe
    path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/"  
    save_path = os.path.join(path, "Subgroup_Analysis_Results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename  = "subgroup_analysis_SF0.csv"
    save_filepath = os.path.join(save_path, filename)

    df.to_csv(save_filepath, header=False, index=False)
    
