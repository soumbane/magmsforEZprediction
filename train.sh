# python train.py /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Lobe_Data/ /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/trained_models_lobe/magms_lobe_smote_train_orig_val_try1.model -b 64 -lr 1e-3 --num_mod 5 -e 20 -exp magms_lobe_smote_train_orig_val_try1.exp --replace_experiment --show_verbose --device cuda:1


python train.py /home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Lobe_Data_exp6/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_exp6.model -b 128 -lr 1e-3 --num_mod 5 -e 50 -exp magms_exp6.exp --replace_experiment --show_verbose --device cuda:0
