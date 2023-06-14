import torch
from ezpred import metrics, Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import metric
from torchmanager_core import view
from typing import Any
import numpy as np

import data

'''y_pred_list_bal: list[torch.Tensor] = []
y_true_list_bal: list[torch.Tensor] = []

y_pred_list_conf: list[torch.Tensor] = []
y_true_list_conf: list[torch.Tensor] = []


@metric
def bal_acc_fn(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    y_pred = input.argmax(dim=-1).cpu().detach().numpy()
    y_pred_list_bal.append(y_pred)
    y_pred_list_bal_np = np.concatenate(y_pred_list_bal).ravel()

    y_true = target.cpu().detach().numpy()
    y_true_list_bal.append(y_true)
    y_true_list_bal_np = np.concatenate(y_true_list_bal).ravel()

    r = balanced_accuracy_score(y_true_list_bal_np, y_pred_list_bal_np)
    print(f"Balanced Accuracy: {r}")
    return torch.tensor(torch.nan)

@metric
def conf_met_func(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    y_pred = input.argmax(dim=-1).cpu().detach().numpy()
    y_pred_list_conf.append(y_pred)
    y_pred_list_conf_np = np.concatenate(y_pred_list_conf).ravel()

    y_true = target.cpu().detach().numpy()
    y_true_list_conf.append(y_true)
    y_true_list_conf_np = np.concatenate(y_true_list_conf).ravel()

    r = confusion_matrix(y_true_list_conf_np, y_pred_list_conf_np)
    print(f"Confusion matrix: {r}")
    return torch.tensor(torch.nan, dtype=torch.float)'''


def test(cfg: TestingConfigs, /) -> dict[str, float]:
    # load dataset
    testing_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, node_num=cfg.node_num)

    # load checkpoint
    if cfg.model.endswith(".model"):
        manager = Manager.from_checkpoint(cfg.model, map_location=cfg.device)
        assert isinstance(manager, Manager), "Checkpoint is not a valid `ezpred.Manager`."
    else:
        raise NotImplementedError(f"Checkpoint {cfg.model} is currently not supported.")
    
    # set up confusion metrics
    conf_met_fn = metrics.ConfusionMetrics(2)
    manager.metric_fns.update({
        # "val_bal_accuracy": bal_acc_fn,
        "conf_met": conf_met_fn
        })

    # manager.metric_fns["val_bal_accuracy"]._eps = 0

    # test checkpoint
    summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)
    return summary


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()
    test(configs)
