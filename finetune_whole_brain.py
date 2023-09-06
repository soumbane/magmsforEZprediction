import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torch.backends import cudnn
from torchmanager_core.typing import Any

import data, ezpred
from ezpred.configs import FinetuningConfigs


def train(cfg: FinetuningConfigs, /, node_num: int = 1) -> Any:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset
    training_dataset = data.DatasetEZ_NodeLevel(cfg.batch_size, cfg.data_dir, node_num=node_num, drop_last=False, mode=data.EZMode.TRAIN, shuffle=True)
    validation_dataset = data.DatasetEZ_NodeLevel(cfg.batch_size, cfg.data_dir, node_num=node_num, mode=data.EZMode.VALIDATE)

    # load checkpoint
    if cfg.pretrained_model.endswith(".model"):
        manager = magnet.Manager.from_checkpoint(cfg.pretrained_model, map_location=cfg.device)
        model = manager.raw_model
    else:
        model = torch.load(cfg.pretrained_model)  # load *.pth

    # extract parameters for sch
    assert isinstance(model, magnet.MAGNET2), "Model is not a valid EzPred MAGNET, the model must be a `magnet.MAGNET2`."
    decoder = model.decoder
    assert isinstance(decoder, torch.nn.Sequential), "Model is not a valid EzPred MAGNET, the decoder must be a `torch.nn.Sequential`."
    sch = decoder[1]

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(sch.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)

    # initialize learning rate scheduler 
    lr_step = max(int(cfg.epochs / 3), 1)  # for 15 epochs (step down every 5 epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5) # reduce lr by half 

    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(cfg.num_mod+1)]
    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean') for _ in range(cfg.num_mod)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(cfg.num_mod)]
    
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

    lr_scheduler_callback = tm.callbacks.LrSchedueler(lr_scheduler, tf_board_writer=experiment_callback.tensorboard.writer) # type:ignore 

    # Final callbacks list
    callbacks_list: list[tm.callbacks.Callback] = [experiment_callback, lr_scheduler_callback]

    # train
    model = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=configs.show_verbose)

    # test
    # save and test with best model on validation dataset  
    checkpoint_path = "experiments/magms_ALL_mod_WB_fold1_node_" + str(node_num) + "_finetuned.exp/checkpoints/best_accuracy.model"
    manager = magnet.Manager.from_checkpoint(checkpoint_path, map_location=cfg.device) 

    # print(f'The best accuracy on validation set occurs at {manager.current_epoch + 1} epoch number') # type:ignore
    
    summary = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus) # type:ignore
    # view.logger.info(summary)
    torch.save(model, cfg.output_model)
    return model, summary['accuracy']


if __name__ == "__main__":
    # get configs
    configs = FinetuningConfigs.from_arguments()
    assert isinstance(configs, FinetuningConfigs)

    # train
    _, acc = train(configs, node_num=configs.node_num)
    print(f"Accuracy for Node num {configs.node_num} is: {acc}")


