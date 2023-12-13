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
import openpyxl


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
    
    ## The actual MAG-MS losses
    # main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(cfg.num_mod+1)]
    # kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean') for _ in range(cfg.num_mod)]
    # mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE() for _ in range(cfg.num_mod)]


    # The MAG-MS losses without any self-distillation
    main_losses: list[magnet.losses.Loss] = [magnet.losses.CrossEntropy() for _ in range(cfg.num_mod+1)]
    main_losses[1] = magnet.losses.CrossEntropy(weight=0) 
    main_losses[2] = magnet.losses.CrossEntropy(weight=0)
    main_losses[3] = magnet.losses.CrossEntropy(weight=0)
    main_losses[4] = magnet.losses.CrossEntropy(weight=0)
    main_losses[5] = magnet.losses.CrossEntropy(weight=0)

    kldiv_losses: list[magnet.losses.Loss] = [magnet.losses.KLDiv(softmax_temperature=3, reduction='batchmean', weight=0) for _ in range(cfg.num_mod)]
    mse_losses: list[magnet.losses.Loss] = [magnet.losses.MSE(weight=0) for _ in range(cfg.num_mod)]

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

    # # test with last model
    # val_summary: dict[str, float] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)

    # view.logger.info(val_summary)
    torch.save(model, cfg.output_model)

    # test with best model on validation dataset
    ckpt_path = os.path.join("experiments", cfg.experiment, "checkpoints", "best_bal_accuracy.model")
    manager = magnet.Manager.from_checkpoint(ckpt_path)

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number') # type:ignore

    # test with best model on validation dataset
    val_summary: dict[str, float] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus) # type:ignore
    view.logger.info(val_summary)

    # return model # type:ignore
    return val_summary['bal_accuracy'], val_summary['sensitivity'], val_summary['specificity'], train_summary['bal_accuracy'], train_summary['sensitivity'], train_summary['specificity'] 


if __name__ == "__main__":
    # get configs
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs)

    base_exp_name = configs.experiment

    num_trials = 5
    
    # Create empty lists to store results for each type (bal_accuracy, sensitivity, specificity)
    val_bal_acc_list = [[] for _ in range(num_trials)]
    val_sen_list = [[] for _ in range(num_trials)]
    val_spec_list = [[] for _ in range(num_trials)]

    train_bal_acc_list = [[] for _ in range(num_trials)]
    train_sen_list = [[] for _ in range(num_trials)]
    train_spec_list = [[] for _ in range(num_trials)]

    # train
    for i in range(num_trials):
        print(f'\n\nStarting Trial {i+1} of Node number {configs.node_num}\n')

        configs.experiment = base_exp_name + "_trial" + str(i+1) + ".exp"

        val_bal_acc, val_sen, val_spec, train_bal_acc, train_sen, train_spec = train(configs) # type:ignore

        val_bal_acc_list[i].append(val_bal_acc) 
        val_sen_list[i].append(val_sen)         
        val_spec_list[i].append(val_spec) 

        train_bal_acc_list[i].append(train_bal_acc) 
        train_sen_list[i].append(train_sen) 
        train_spec_list[i].append(train_spec) 

    # Combine data
    row_data_val = [configs.node_num] + [val_bal_acc_list[j][0] for j in range(num_trials)] + [val_sen_list[j][0] for j in range(num_trials)] + [val_spec_list[j][0] for j in range(num_trials)] 

    row_data_train = [configs.node_num] + [train_bal_acc_list[j][0] for j in range(num_trials)] + [train_sen_list[j][0] for j in range(num_trials)] + [train_spec_list[j][0] for j in range(num_trials)]

    # Create a DataFrame
    headers_val = ['Node #', 'Val_Balanced_Accuracy_1', 'Val_Balanced_Accuracy_2', 'Val_Balanced_Accuracy_3', 'Val_Balanced_Accuracy_4', 'Val_Balanced_Accuracy_5', 'Val_sensitivity_1', 'Val_sensitivity_2', 'Val_sensitivity_3', 'Val_sensitivity_4', 'Val_sensitivity_5', 'Val_specificity_1', 'Val_specificity_2', 'Val_specificity_3', 'Val_specificity_4', 'Val_specificity_5']
    
    headers_train = ['Node #', 'Train_Balanced_Accuracy_1', 'Train_Balanced_Accuracy_2', 'Train_Balanced_Accuracy_3', 'Train_Balanced_Accuracy_4', 'Train_Balanced_Accuracy_5', 'Train_sensitivity_1', 'Train_sensitivity_2', 'Train_sensitivity_3', 'Train_sensitivity_4', 'Train_sensitivity_5', 'Train_specificity_1', 'Train_specificity_2', 'Train_specificity_3', 'Train_specificity_4', 'Train_specificity_5']

    df_val = pd.DataFrame([row_data_val], columns=headers_val)

    df_train = pd.DataFrame([row_data_train], columns=headers_train)

    # Saving to Excel
    path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/"
    save_path = os.path.join(path, "Node_"+str(configs.node_num), "Results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # filename_val = "results_val.xlsx"
    filename_val = "results_val_NO_Distillation.xlsx"
    save_filepath_val = os.path.join(save_path, filename_val)

    df_val.to_excel(save_filepath_val, index=False, sheet_name='Sheet1')

    # filename_train = "results_train.xlsx"
    filename_train = "results_train_NO_Distillation.xlsx"
    save_filepath_train = os.path.join(save_path, filename_train)

    df_train.to_excel(save_filepath_train, index=False, sheet_name='Sheet1')

    print("\nDone!")

    ###############################################################################
    # # get configs - perform only 1 or 2 trials to modify the experiments for trial 
    # configs = TrainingConfigs.from_arguments()
    # assert isinstance(configs, TrainingConfigs)

    # base_exp_name = configs.experiment

    # num_trials = [configs.trial_num] # the trial you want to try minus 1 (e.g. if you want to try trial 1, then put 0 here)
    # if num_trials[0] + 1 == 1:
    #     select_data_bal_acc = 'B2'
    #     select_data_sen = 'G2'
    #     select_data_spec = 'L2'
    # elif num_trials[0] + 1 == 2:
    #     select_data_bal_acc = 'C2'
    #     select_data_sen = 'H2'
    #     select_data_spec = 'M2'
    # elif num_trials[0] + 1 == 3:
    #     select_data_bal_acc = 'D2'
    #     select_data_sen = 'I2'
    #     select_data_spec = 'N2'
    # elif num_trials[0] + 1 == 4:
    #     select_data_bal_acc = 'E2'
    #     select_data_sen = 'J2'
    #     select_data_spec = 'O2'
    # elif num_trials[0] + 1 == 5:
    #     select_data_bal_acc = 'F2'
    #     select_data_sen = 'K2'
    #     select_data_spec = 'P2'
    # else:
    #     raise ValueError('The trial number is not valid!')
    
    # # Load the excel files - both for validation and training
    # path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/"
    # load_path = os.path.join(path, "Node_"+str(configs.node_num), "Results")

    # # Load validation file
    # filename_val = "results_val.xlsx"
    # load_filepath_val = os.path.join(load_path, filename_val)

    # val_file = openpyxl.load_workbook(load_filepath_val)
    # val_sheet = val_file['Sheet1']

    # # Load Training file
    # filename_train = "results_train.xlsx"
    # load_filepath_train = os.path.join(load_path, filename_train)

    # train_file = openpyxl.load_workbook(load_filepath_train)
    # train_sheet = train_file['Sheet1']

    # # train
    # for i in num_trials:
    #     print(f'\n\nStarting Trial {i+1} of Node number {configs.node_num}\n')

    #     configs.experiment = base_exp_name + "_trial" + str(i+1) + ".exp"

    #     val_bal_acc, val_sen, val_spec, train_bal_acc, train_sen, train_spec = train(configs) # type:ignore

    #     val_sheet[select_data_bal_acc] = val_bal_acc  # type:ignore
    #     val_sheet[select_data_sen] = val_sen  # type:ignore
    #     val_sheet[select_data_spec] = val_spec  # type:ignore

    #     train_sheet[select_data_bal_acc] = train_bal_acc  # type:ignore
    #     train_sheet[select_data_sen] = train_sen  # type:ignore
    #     train_sheet[select_data_spec] = train_spec  # type:ignore
    
    # save_path = os.path.join(path, "Node_"+str(configs.node_num), "Results")
    
    # # Save modified validation file
    # filename_val = "results_val.xlsx"
    # save_filepath_val = os.path.join(save_path, filename_val)
    # val_file.save(save_filepath_val)
    # val_file.close()

    # # Save modified training file
    # filename_train = "results_train.xlsx"
    # save_filepath_train = os.path.join(save_path, filename_train)
    # train_file.save(save_filepath_train)
    # train_file.close()

    # print("\nDone!")

