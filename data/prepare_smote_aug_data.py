# Use SMOTE to augment EZ predict dataset to balance both classes and create more samples
import os
import numpy as np
from collections import Counter
from scipy.io import loadmat, savemat
from imblearn.over_sampling import SMOTE, ADASYN


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
        X_mat_RI = X_mat_l[RI_mat_name] # RI matrix: 1x1400
    
        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_RI).any()):  
            X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

        """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
        X_mat_lconn = loadmat(raw_path_Conn)
        X_mat_DWIC = X_mat_lconn[Conn_mat_name]  # DWIC matrix: 1x499
                    
        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_DWIC).any()):
            X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

        X_combined_1 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features

        """Load the Label Matrix from .mat files.""" 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[label_mat_name]
        Y_mat_aug_1 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

        return X_combined_1, Y_mat_aug_1

    elif mode == "valid":
        path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')
        RI_file = f"ValidCohort_NonEZvsEZ_RI_node{node_num}_ALL.mat"
        Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
        label_file = f"ValidCohort_NonEZvsEZ_label_node{node_num}_ALL.mat"
        RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
        Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
        label_mat_name = "ValidCohort_NonEZvsEZ_label"

        raw_path_RI = os.path.join(root,path,RI_file)
        raw_path_Conn = os.path.join(root,path,Conn_file)
        raw_path_label = os.path.join(root,path,label_file)

        """Load the Relative Intensity (RI) Data Matrix from .mat files.""" 
        X_mat_l = loadmat(raw_path_RI)
        X_mat_RI = X_mat_l[RI_mat_name] # RI matrix: 1x1400

        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_RI).any()):  
            X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

        """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
        X_mat_lconn = loadmat(raw_path_Conn)
        X_mat_DWIC = X_mat_lconn[Conn_mat_name]  # DWIC matrix: 1x499
                    
        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_DWIC).any()):
            X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

        X_combined_2 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features

        """Load the Label Matrix from .mat files.""" 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[label_mat_name]
        Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

        return X_combined_2, Y_mat_aug_2

    else:
        raise NotImplementedError(f"Mode must be either model or valid to load dataset.")

def augment_data(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5, num_samples: int = 100, random_state: int = 100):

    # sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state, sampling_strategy={0:num_samples, 1:num_samples}) # type:ignore
    sm = ADASYN(n_neighbors=k_neighbors, random_state=random_state, sampling_strategy={0:num_samples, 1:num_samples}) # type:ignore
    X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore
    
    return X_aug, Y_aug

def main(root: str, node_num: str, num_aug_samples: int = 100, k_neighbors: int = 5):
    X_model, Y_model = load_data(root, node_num, mode="model")  # type:ignore

    X_valid, Y_valid = load_data(root, node_num, mode="valid")  # type:ignore

    # concatenate model and valid to create all patients data
    X_combined = np.concatenate((X_model, X_valid), axis=0) 
    Y_combined = np.concatenate((Y_model, Y_valid), axis=0)

    # augment all patients data using SMOTE (balance dataset and create more samples)
    X_aug, Y_aug = augment_data(X_combined, Y_combined, k_neighbors = k_neighbors, num_samples = num_aug_samples, random_state=100)

    # randomly shuffle (with seed) data
    np.random.seed(0)
    np.random.shuffle(X_aug)

    np.random.seed(0)
    np.random.shuffle(Y_aug)

    # divide data into training and testing
    x_train_len = int(0.8*len(X_aug))

    X_aug_train = X_aug[:x_train_len,:] # type:ignore
    Y_aug_train = Y_aug[:x_train_len]

    X_aug_test = X_aug[x_train_len:,:] # type:ignore
    Y_aug_test = Y_aug[x_train_len:]

    print(X_aug_train.shape)
    print(Y_aug_train.shape)
    print('Resampled Y_train shape %s' % Counter(Y_aug_train))

    print(X_aug_test.shape)
    print(Y_aug_test.shape)
    print('Resampled Y_test shape %s' % Counter(Y_aug_test))

    # save the augmented data
    savemat('X_train_aug_node948.mat', {"X_aug_train":X_aug_train})
    savemat('Y_train_aug_node948.mat', {"Y_aug_train":Y_aug_train})

    savemat('X_test_aug_node948.mat', {"X_aug_test":X_aug_test})
    savemat('Y_test_aug_node948.mat', {"Y_aug_test":Y_aug_test})


if __name__ == "__main__":

    # Root Folder
    root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'
    node_num = "948"

    main(root, node_num, num_aug_samples=6000, k_neighbors=3)

