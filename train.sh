python train.py /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Lobe_Data_exp9/ /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/trained_models_lobe/magms_exp9.model -b 8 -lr 1e-3 --num_mod 5 -e 1 -exp magms_exp9.exp --replace_experiment --show_verbose --device cuda:1


python train.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Lobe_Data_exp8/ /home/user1/Desktop/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_exp8.model -b 128 -lr 1e-3 --num_mod 5 --train_mod ALL -e 50 -exp magms_exp8.exp --replace_experiment --show_verbose --device cuda:3
