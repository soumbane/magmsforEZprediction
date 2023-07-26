import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torch.backends import cudnn

import data, ezpred
from ezpred.configs import FinetuningConfigs


def train(cfg: FinetuningConfigs, /) -> magnet.MAGNET2:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset
    training_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, drop_last=True, mode=data.EZMode.TRAIN, shuffle=True)
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE)
    testing_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST)

    # load checkpoint
    if cfg.pretrained_model.endswith(".model"):
        manager = magnet.Manager.from_checkpoint(cfg.pretrained_model, map_location=cfg.device)
        model = manager.raw_model
    else:
        model = torch.load(cfg.pretrained_model)

    # extract parameters for sch
    assert isinstance(model, magnet.MAGNET2), "Model is not a valid EzPred MAGNET, the model must be a `magnet.MAGNET2`."
    decoder = model.decoder
    assert isinstance(decoder, torch.nn.Sequential), "Model is not a valid EzPred MAGNET, the decoder must be a `torch.nn.Sequential`."
    sch = decoder[1]

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(sch.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
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
    configs = FinetuningConfigs.from_arguments()
    assert isinstance(configs, FinetuningConfigs)

    # train
    train(configs)
