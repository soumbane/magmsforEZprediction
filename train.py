import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torch.backends import cudnn
import os

import data, ezpred
from ezpred.configs import TrainingConfigs


def train(cfg: TrainingConfigs, /) -> magnet.MAGNET2:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset for whole brain
    training_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, drop_last=True, mode=data.EZMode.TRAIN, fold_no=cfg.fold_no, shuffle=True)
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.VALIDATE, fold_no=cfg.fold_no)
    
    # build model
    model = ezpred.build(2, train_modality=cfg.train_mod)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)

    # initialize learning rate scheduler 
    lr_step = max(int(cfg.epochs / 5), 1)  # for 25 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5) # reduce lr by half     
    
    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(cfg.num_mod+1)]
    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean') for _ in range(cfg.num_mod)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(cfg.num_mod)]
    magms_loss = magnet.losses.MAGMSLoss(main_losses, distillation_loss=kldiv_losses, feature_losses=mse_losses)
    
    # if we want to track only the validation accuracy
    # metric_fns: dict[str, tm.metrics.Metric] = {
    #     "CE_loss_all": main_losses[0],
    #     "val_accuracy": tm.metrics.SparseCategoricalAccuracy(),
    #     "val_bal_accuracy": ezpred.metrics.BalancedAccuracyScore()
    # }

    # if we want to track both the training and validation accuracy
    metric_fns: dict[str, tm.metrics.Metric] = {
        "CE_loss_all": main_losses[0],
        "accuracy": tm.metrics.SparseCategoricalAccuracy(),
        "bal_accuracy": ezpred.metrics.BalancedAccuracyScore(),
        "sensitivity": ezpred.metrics.SensitivityScore(),
        "specificity": ezpred.metrics.SpecificityScore(),
    }

    # compile manager
    manager = magnet.Manager(model, optimizer=optimizer, loss_fn=magms_loss, metrics=metric_fns)

    # initialize callbacks
    experiment_callback = tm.callbacks.Experiment(cfg.experiment, manager, monitors=["accuracy", "bal_accuracy", "sensitivity"])    
    
    lr_scheduler_callback = tm.callbacks.LrSchedueler(lr_scheduler, tf_board_writer=experiment_callback.tensorboard.writer) # type:ignore   

    # Final callbacks list
    callbacks_list: list[tm.callbacks.Callback] = [experiment_callback, lr_scheduler_callback]

    # train
    model = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=configs.show_verbose)

    # test
    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    view.logger.info(summary)
    torch.save(model, cfg.output_model)
    return model


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs)

    # train
    train(configs)
