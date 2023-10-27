import magnet, torch, torchmanager as tm
from torchmanager_core import view
from torchmanager.metrics import BinaryConfusionMetric, metric
from torch.backends import cudnn
import os

import data, ezpred
from ezpred.nn.fusion import FusionType
from ezpred.configs import TrainingConfigs
from typing import Any, Union, Tuple
import pandas as pd


# def train(cfg: TrainingConfigs, /) -> magnet.MAGNET2:
def train(cfg: TrainingConfigs, /) -> Union[magnet.MAGNET2, Tuple[float, float, float, float, float, float]]:
    # initialize seed for reproducibility
    if cfg.seed is not None:
        from torchmanager_core import random
        random.freeze_seed(cfg.seed)
        cudnn.benchmark = False 
        cudnn.deterministic = True  

    # initialize dataset for whole brain
    training_dataset = data.DatasetEZ_Node(cfg.batch_size, cfg.data_dir, drop_last=False, mode=data.EZMode.TRAIN, shuffle=True, node_num=str(cfg.node_num))
    validation_dataset = data.DatasetEZ_Node(cfg.batch_size, cfg.data_dir, mode=data.EZMode.TEST, node_num=str(cfg.node_num))
    
    model = ezpred.build(2, out_main_ch=64, out_filters=64, filters_t1=[8,16,32], filters_t2=[8,16,32], filters_flair=[8,16,32], filters_dwi=[16,32,64], filters_dwic=[8,16,32], main_downsample=True, filters_shfe = [64,128], fusion=FusionType.MID_MEAN, train_modality=cfg.train_mod) 

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

    # train and test on validation data
    model, train_summary = manager.fit(training_dataset, epochs=cfg.epochs, val_dataset=validation_dataset, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=configs.show_verbose, return_summary=True) # type:ignore

    # test with last model
    val_summary: dict[str, float] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    view.logger.info(val_summary)
    torch.save(model, cfg.output_model)

    # test with best model on validation dataset  
    manager = magnet.Manager.from_checkpoint("experiments/magms_node_888_tr_1.exp/checkpoints/best_bal_accuracy.model")

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number') # type:ignore

    val_summary: dict[str, float] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus) # type:ignore
    view.logger.info(val_summary)

    # return model # type:ignore
    return val_summary['bal_accuracy'], val_summary['sensitivity'], val_summary['specificity'], train_summary['bal_accuracy'], train_summary['sensitivity'], train_summary['specificity'] 


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs)

    # train
    # train(configs)

    val_balanced_acc = [] 
    train_balanced_acc = []

    val_sensitivity = []
    train_sensitivity = []

    val_specificity = []
    train_specificity = []

    # train
    for i in range(5):
        print(f'Trial: {i}')
        val_bal_acc, val_sen, val_spec, train_bal_acc, train_sen, train_spec = train(configs) # type:ignore
        val_balanced_acc.append(val_bal_acc)
        val_sensitivity.append(val_sen)
        val_specificity.append(val_spec)

        train_balanced_acc.append(train_bal_acc)
        train_sensitivity.append(train_sen)
        train_specificity.append(train_spec)

    # dictionary of lists for validation results
    val_balanced_acc_dict = {'Val_Balanced_Accuracy': val_balanced_acc, 'Val_sensitivity': val_sensitivity, 'Val_specificity': val_specificity}    

    df_val = pd.DataFrame(val_balanced_acc_dict) # dataframe for validation results

    # dictionary of lists for training results
    train_balanced_acc_dict = {'Train_Balanced_Accuracy': train_balanced_acc, 'Train_sensitivity': train_sensitivity, 'Train_specificity': train_specificity}    

    df_train = pd.DataFrame(train_balanced_acc_dict) # dataframe for training results  

    # saving the dataframes
    # path = "/home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/"  
    path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/"  
    save_path = os.path.join(path, "Node_"+str(configs.node_num), "Results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename  = "Accuracy_results.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df_val.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    df_train.to_excel(save_filepath, sheet_name='Sheet2', header=True, index=False)

    print("Done!")
