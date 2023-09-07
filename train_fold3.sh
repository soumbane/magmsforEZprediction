python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_T1_DWIC_WB_fold3.model -b 32 -lr 5e-4 --num_mod 5 --fold_no 3 --train_mod T1-DWIC -e 25 -exp magms_T1_DWIC_WB_fold3.exp --replace_experiment --show_verbose --device cuda:2

python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_T1_T2_DWIC_WB_fold3.model -b 32 -lr 5e-4 --num_mod 5 --fold_no 3 --train_mod T1-T2-DWIC -e 25 -exp magms_T1_T2_DWIC_WB_fold3.exp --replace_experiment --show_verbose --device cuda:2

python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_T1_T2_DWI_DWIC_WB_fold3.model -b 32 -lr 5e-4 --num_mod 5 --fold_no 3 --train_mod T1-T2-DWI-DWIC -e 25 -exp magms_T1_T2_DWI_DWIC_WB_fold3.exp --replace_experiment --show_verbose --device cuda:2


python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_T1_T2_DWI_DWIC_WB_fold5.model -b 32 -lr 5e-4 --num_mod 5 --fold_no 5 --train_mod T1-T2-DWI-DWIC -e 25 -exp magms_T1_T2_DWI_DWIC_WB_fold5.exp --replace_experiment --show_verbose --device cuda:2
