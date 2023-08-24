# Use SMOTE to augment EZ predict dataset to balance both classes and create more samples for the whole brain (comibining all the nodes)
import os
import numpy as np
from collections import Counter
from scipy.io import loadmat, savemat


def z_score_norm(X: np.ndarray):

    X_mean = np.mean(X, axis=0, dtype=np.float64)
    X_std = np.std(X, axis=0, dtype=np.float64)

    X_norm = (X - X_mean)/X_std

    # check for NaN values and replace NaN values with 0
    if (np.isnan(X_norm).any()):
        X_norm = np.nan_to_num(X_norm, nan=0) 
    
    return X_norm


def save_control_data_as_separate_nodes(save_dir: str, X: np.ndarray, Y: np.ndarray, mode: str = "valid") -> None:
    
    for i in range(len(Y)):
        if mode == "train":
            raise NotImplementedError("Train data not saved for control patients.")

        elif mode == "valid":
            savemat(os.path.join(save_dir,'X_val_control_whole_brain_node' + str(i) + '.mat'), {"X_control_valid_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_val_control_whole_brain_node' + str(i) + '.mat'), {"Y_control_valid_node" + str(i):Y[i]})

        else:
            raise KeyError(f"The mode must be either train, valid.")


def main(root: str):    
    
    # Load the control data of all nodes of 2 patients
    path = os.path.join(root,'Control_Data')                
    X_file = f"Controls_ALL.mat"
    Y_file = f"Controls_Label_ALL.mat"    
    X_mat_name = "Controls_ALL"
    Y_mat_name = "Controls_Label_ALL"
    
    raw_path_X = os.path.join(path,X_file)
    raw_path_Y = os.path.join(path,Y_file)

    """Load the X Matrix from .mat files.""" 
    X_mat_l = loadmat(raw_path_X)
    X_control_patients = X_mat_l[X_mat_name]

    """Load the Y Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_Y)
    Y_control_patients = Y_mat_l[Y_mat_name]

    # Perform z-score normalization across patients
    X_control_patients_norm = z_score_norm(X_control_patients)  # type:ignore
    print(f"X_all_patients max: {np.max(X_control_patients_norm)}")
    print(f"X_all_patients min: {np.min(X_control_patients_norm)}")
    Y_control_patients = Y_control_patients.reshape(Y_control_patients.shape[0])

    print('UnAugmented Y_all_patients shape %s' % Counter(Y_control_patients))

    X_val = X_control_patients_norm # With z-score normalization
    # X_val = X_control_patients # NO z-score normalization
    Y_val = Y_control_patients
        
    # save each node of the control data separately
    save_dir_val = 'Val_NonEZvsEZ_whole_brain_control_separate'
    if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)

    save_control_data_as_separate_nodes(save_dir_val, X_val, Y_val, mode="valid")


if __name__ == "__main__":

    # Root Folder
    root='/home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/'
    # root='/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsEZpred/'

    main(root)


