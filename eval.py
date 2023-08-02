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


def test(cfg: TestingConfigs, /) -> dict[str, float]:
    # load dataset
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE)
    # testing_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST)
    
    # validation_dataset = data.DatasetEZ_NodeLevel(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE)

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
    manager.target_dict = {
        0: "T1",
        1: "T2",
        2: "FLAIR",
        3: "DWI",
        4: "DWIC",
    }

    print(f'The best accuracy on validation set occurs at {manager.current_epoch + 1} epoch number')

    # test checkpoint with validation dataset
    summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)
    
    # test checkpoint with testing dataset
    # summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    # if conf_met_fn.results is not None:
    #     summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    return summary


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()
    test(configs)
