import magnet, torch, torchmanager as tm

import data, ezpred
from ezpred.configs import TrainingConfigs


def train(cfg: TrainingConfigs, /) -> magnet.MAGNET2:
    # initialize dataset
    training_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, drop_last=True, mode=data.EZMode.TRAIN, shuffle=True, node_num=cfg.node_num)
    validation_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, node_num=cfg.node_num)
    testing_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, node_num=cfg.node_num)

    # build model
    model = ezpred.build(1, 2)

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(5)]
    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv() for _ in range(5)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(5)]
    magms_loss = magnet.losses.MAGMSLoss(main_losses, kldiv_losses, mse_losses)
    metric_fns: dict[str, tm.metrics.Metric] = {
        "accuracy": tm.metrics.SparseCategoricalAccuracy(),
    }

    # compile manager
    manager = magnet.Manager(model, optimizer=optimizer, loss_fn=magms_loss, metrics=metric_fns)

    # initialize callbacks
    experiment_callback = tm.callbacks.Experiment(cfg.experiment, manager, monitors=)