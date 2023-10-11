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
    training_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, drop_last=True, mode=data.EZMode.TRAIN, fold_no=cfg.fold_no, shuffle=True)
    validation_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, fold_no=cfg.fold_no)

    # testing_dataset = data.DatasetEZ_WB(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, fold_no=cfg.fold_no)
    
    # build model
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=128, filters_t1=[32,64,128], filters_t2=[32,64,128], filters_flair=[32,64,128], filters_dwi=[32,64,128], filters_dwic=[32,64,128], main_downsample=True, filters_shfe = [128,256], fusion=FusionType.MID_MEAN)

    # worked for node 948, 917 with the referenced seeds
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=32, filters_t1=[8,16], filters_t2=[8,16], filters_flair=[8,16], filters_dwi=[16,32], filters_dwic=[8,16], main_downsample=True, filters_shfe = [32,64], fusion=FusionType.MID_MEAN) # bal_acc of 70 (948) & 75 (917) (batch-size:256)

    # try for node 917 & 948
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[16,32,64], filters_t2=[16,32,64], filters_flair=[16,32,64], filters_dwi=[32,64,128], filters_dwic=[16,32,64], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 77.5 (948) & 69.44 (917) (batch-size:256)

    # try for node 916
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[16,32,64], filters_t2=[16,32,64], filters_flair=[16,32,64], filters_dwi=[32,64,128], filters_dwic=[16,32,64], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 73.95 (batch-size:128)

    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=32, filters_t1=[16,32], filters_t2=[16,32], filters_flair=[16,32], filters_dwi=[32,64], filters_dwic=[16,32], main_downsample=True, filters_shfe = [32,64], fusion=FusionType.MID_MEAN) # node 916:bal_acc of 64.2 (batch-size:128)

    ##########################################################################################
    ## Dong Approach - Node level

    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=32, out_filters=32, filters_t1=[8,16], filters_t2=[8,16], filters_flair=[8,16], filters_dwi=[16,32], filters_dwic=[8,16], main_downsample=True, filters_shfe = [32,64], fusion=FusionType.MID_MEAN) # bal_acc of 71.43 (916) &  (917) (batch-size:4)

    # # try for node 916
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=16, out_filters=16, filters_t1=[4,8], filters_t2=[4,8], filters_flair=[4,8], filters_dwi=[8,16], filters_dwic=[4,8], main_downsample=True, filters_shfe = [16,32], fusion=FusionType.MID_MEAN) # bal_acc of 78.57 (916) &  (917) (batch-size:4)

    # try for node 916
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[8,16,32], filters_t2=[8,16,32], filters_flair=[8,16,32], filters_dwi=[16,32,64], filters_dwic=[8,16,32], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 71.43 (916) &  (917) (batch-size:4)

    ##########################################################################################
    ## Dong Approach - ROI level

    # try for Inferior Temporal ROI (Node 916-931) - 16 nodes
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[16,32,64], filters_t2=[16,32,64], filters_flair=[16,32,64], filters_dwi=[16,32,64], filters_dwic=[16,32,64], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 62.22 (batch-size:16)

    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[8,16,32], filters_t2=[8,16,32], filters_flair=[8,16,32], filters_dwi=[16,32,64], filters_dwic=[8,16,32], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of  (batch-size:16)

    ##########################################################################################

    ## Dong Approach - Lobe level

    # try for Temporal Lobe (Node 888-983) - 94 nodes
    # model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[16,32,64], filters_t2=[16,32,64], filters_flair=[16,32,64], filters_dwi=[16,32,64], filters_dwic=[16,32,64], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 61.61 (batch-size:32)

    ##########################################################################################

    ## Dong Approach - ROI level - per node aug

    # try for Inferior Temporal ROI (Node 916-931) - 16 nodes
    model = ezpred.build(2, train_modality=cfg.train_mod, out_main_ch=64, out_filters=64, filters_t1=[8,16,32], filters_t2=[8,16,32], filters_flair=[8,16,32], filters_dwi=[16,32,64], filters_dwic=[8,16,32], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN) # bal_acc of 62.22 (batch-size:16)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)

    # initialize learning rate scheduler 
    # lr_step = max(int(cfg.epochs / 5), 1)  # for 50 epochs
    lr_step = max(int(cfg.epochs / 2), 1)  # for 30 epochs
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
    
    lr_scheduler_callback = tm.callbacks.LrSchedueler(lr_scheduler, tf_board_writer=experiment_callback.tensorboard.writer) # type:ignore   

    # Final callbacks list
    callbacks_list: list[tm.callbacks.Callback] = [experiment_callback]
    # callbacks_list: list[tm.callbacks.Callback] = [experiment_callback, lr_scheduler_callback]

    # # train and test on validation data
    model = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=configs.show_verbose)

    # test with last model
    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    view.logger.info(summary)
    torch.save(model, cfg.output_model)

    # test with best model on validation dataset  
    manager = magnet.Manager.from_checkpoint("experiments/magms_exp13.exp/checkpoints/best_bal_accuracy.model")

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    view.logger.info(summary)

    return model


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs)

    # train
    train(configs)
