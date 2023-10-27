import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torchmanager.metrics import BinaryConfusionMetric, metric
from torch.backends import cudnn
import os

import data, ezpred
from ezpred.nn.fusion import FusionType
from ezpred.configs import TrainingConfigs


def train(cfg: TrainingConfigs, /) -> magnet.MAGNET2:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset for whole brain
    training_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, drop_last=False, mode=data.EZMode.TRAIN, shuffle=True)
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST)
    
    model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[8,16,32], filters_t2=[8,16,32], filters_flair=[8,16,32], filters_dwi=[16,32,64], filters_dwic=[8,16,32], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) 

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)  
    
    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(cfg.num_mod+1)]
    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean') for _ in range(cfg.num_mod)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(cfg.num_mod)]
    magms_loss = magnet.losses.MAGMSLoss(main_losses, distillation_loss=kldiv_losses, feature_losses=mse_losses)

    # if we want to track both the training and validation accuracy
    metric_fns: dict[str, tm.metrics.Metric] = {
        "CE_loss_all": main_losses[0],
        "accuracy": ezpred.metrics.AccuracyScore(),
        "bal_accuracy": ezpred.metrics.BalancedAccuracyScore(),
        "sensitivity": ezpred.metrics.SensitivityScore(),
        "specificity": ezpred.metrics.SpecificityScore(),
    }

    for m in metric_fns.values():
        if isinstance(m, BinaryConfusionMetric):
            m._class_index = 0 # since we consider non-EZ (class 0) as positive class

    # compile manager
    manager = magnet.Manager(model, optimizer=optimizer, loss_fn=magms_loss, metrics=metric_fns)

    # initialize callbacks
    experiment_callback = tm.callbacks.Experiment(cfg.experiment, manager, monitors=["accuracy", "bal_accuracy", "sensitivity", "specificity"])      

    # Final callbacks list
    callbacks_list: list[tm.callbacks.Callback] = [experiment_callback]

    # # train and test on validation data
    model = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=configs.show_verbose)

    # test with last model
    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    view.logger.info(summary)
    torch.save(model, cfg.output_model)

    # test with best model on validation dataset  
    manager = magnet.Manager.from_checkpoint("experiments/magms_exp10.exp/checkpoints/best_bal_accuracy.model")

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number') # type:ignore

    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus) # type:ignore
    view.logger.info(summary)

    return model # type:ignore


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs)

    # train
    train(configs)
