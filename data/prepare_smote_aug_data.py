# Use SMOTE to augment EZ predict dataset to balance both classes and create more samples
import os
import random
import numpy as np
from collections import Counter
from scipy.io import loadmat

import torch
import torch.nn as nn

from imblearn.over_sampling import SMOTE

def load_data(root: str, node_num: str, mode: str = "model"):

    if mode == "model":
        # initialize path
        path = os.path.join(root,'Valid_NonEZvsEZ_ALL')
        RI_file = f"Valid_NonEZvsEZ_RI_node{node_num}_ALL.mat"
        Conn_file = f"Valid_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
        label_file = f"Valid_NonEZvsEZ_label_node{node_num}_ALL.mat"
        RI_mat_name = "ModelCohort_NonEZvsEZ_RI"
        Conn_mat_name = "ModelCohort_NonEZvsEZ_Conn"
        label_mat_name = "ModelCohort_NonEZvsEZ_label"


        raw_path_RI = os.path.join(root,path,RI_file)
        raw_path_Conn = os.path.join(root,path,Conn_file)
        raw_path_label = os.path.join(root,path,label_file)

        """Load the Relative Intensity (RI) Data Matrix from .mat files.""" 
        X_mat_l = loadmat(raw_path_RI)
        X_mat_RI = X_mat_l[RI_mat_name]
        # print(f'RI Feature Matrix Shape: {X_mat_RI.shape}')

        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_RI).any()):  
            X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

        """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
        X_mat_lconn = loadmat(raw_path_Conn)
        X_mat_DWIC = X_mat_lconn[Conn_mat_name]  # DWIC matrix: 1x499
        # print(f'DWIC Connectome profile matrix shape: {X_mat_DWIC.shape}')
                    
        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_DWIC).any()):
            X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

        # X_combined = [X_mat_T1, X_mat_T2, X_mat_FLAIR, X_mat_DWI, X_mat_DWIC]
        X_combined_1 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features
        # print(f'X_combined matrix shape: {X_combined.shape}')

        """Load the Label Matrix from .mat files.""" 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[label_mat_name]
        # print(f'GT-Labels shape:{Y_mat_aug.shape}')
        Y_mat_aug_1 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
        # Y_label = torch.from_numpy(Y_mat_aug) # for CrossEntropyLoss

        print('Resampled dataset shape %s' % Counter(Y_mat_aug_1))

        return X_combined_1, Y_mat_aug_1

def main(root: str, node_num: str):
    X_model, Y_model = load_data(root, node_num, mode="model")  # type:ignore

    print(X_model.shape)
    print(Y_model.shape)


if __name__ == "__main__":
    # Root Folder
    root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'
    node_num = "948"
    main(root, node_num)

