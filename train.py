import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torch.backends import cudnn

import data, ezpred
from ezpred.configs import TrainingConfigs


def train(cfg: TrainingConfigs, /) -> magnet.MAGNET2:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset
    training_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, drop_last=True, mode=data.EZMode.TRAIN, shuffle=True, node_num=cfg.node_num)
    validation_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, node_num=cfg.node_num)
    testing_dataset = data.DatasetEZ(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, node_num=cfg.node_num)

    # build model
    model = ezpred.build(1, 2)

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(6)]
    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean') for _ in range(5)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(5)]
    magms_loss = magnet.losses.MAGMSLoss(main_losses, distillation_loss=kldiv_losses, feature_losses=mse_losses)
    metric_fns: dict[str, tm.metrics.Metric] = {
        "CE_loss_all": main_losses[0],
        "val_accuracy": tm.metrics.SparseCategoricalAccuracy(),
        "val_bal_accuracy": ezpred.metrics.BalancedAccuracyScore()
    }

    # compile manager
    manager = magnet.Manager(model, optimizer=optimizer, loss_fn=magms_loss, metrics=metric_fns)

    # initialize callbacks
    experiment_callback = tm.callbacks.Experiment(cfg.experiment, manager, monitors=["accuracy", "bal_accuracy"])
    # early_stop = tm.callbacks.EarlyStop("bal_accuracy", steps=20)

    # train
    model = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=[experiment_callback], show_verbose=configs.show_verbose)

    # test
    summary = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    view.logger.info(summary)
    torch.save(model, cfg.output_model)
    return model


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()

    # train
    train(configs)
