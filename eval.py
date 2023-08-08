import torch
from ezpred import metrics
from magnet import Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import metric
from torchmanager_core import view
from typing import Any
import numpy as np

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
    # load dataset
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE)
    # testing_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST)    
    
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
    # summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=False, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    
    # test checkpoint with testing dataset
    # summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    # if conf_met_fn.results is not None:
    #     summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    return summary['accuracy'], manager.target_dict


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    accuracy = []

    for i in range(1,32):
        dict_mod = get_target_dict(i)    
        acc, mod_dict = test(configs, target_dict=dict_mod)
        accuracy.append(acc)
        print(f"Testing modality combination: {mod_dict}, accuracy is: {acc}\n")

    print(f"Final Testing modality combination mean is: {np.mean(accuracy)}")
    