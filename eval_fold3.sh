python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_FLAIR_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1

python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_T2_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1

python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_T2_FLAIR_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1

python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_T1_FLAIR_DWIC_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1

python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_T1_FLAIR_DWI_DWIC_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1

python eval.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/magms_T1_T2_FLAIR_DWIC_WB_fold3.exp/checkpoints/best_accuracy.model -b 32 --fold_no 3 --replace_experiment --device cuda:1
