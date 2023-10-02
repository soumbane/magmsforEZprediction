python train.py /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Lobe_Data/ /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/trained_models_lobe/magms_lobe_smote_train_orig_val_try.model -b 96 -lr 1e-3 --num_mod 5 -e 2 -exp magms_lobe_smote_train_orig_val_try.exp --replace_experiment --show_verbose --device cuda:0


# python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Lobe_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_nodebynode_smote_train_orig_val_WB_fold1.model -b 42 -lr 5e-4 --num_mod 5 --fold_no 1 -e 25 -exp magms_nodebynode_smote_train_orig_val_WB_fold1.exp --replace_experiment --show_verbose --device cuda:0
