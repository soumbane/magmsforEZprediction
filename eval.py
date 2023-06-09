from ezpred import metrics, Manager
from ezpred.configs import TestingConfigs
from torchmanager_core import view
from typing import Any

import data


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
    manager.metric_fns.update({"conf_met": conf_met_fn})

    # test checkpoint
    summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    summary.update({"conf_met": conf_met_fn.results})
    view.logger.info(summary)
    return summary


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()
    test(configs)
