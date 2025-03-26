import os
from typing import Any, Union
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM

from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt


def calculate_metrics(y_pred, y_true):
    return balanced_accuracy_score(y_true, y_pred)

root='/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/'

def get_list_of_node_nums():
    node_numbers_with_smote = [
    "504", "506", "508", "509", "510"
    ]

    return node_numbers_with_smote

right_hemis_nodes = get_list_of_node_nums()
print(len(right_hemis_nodes))

def get_modality(num: int, X: np.ndarray, fill_zeros: bool = False) -> np.ndarray:
    if num == 1:
        dict_mod = {4:"DWIC"}
        list_mod = [0,0,0,0,1]
        if fill_zeros:
            X_modality = X[:, 1400:]
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = X[:, 1400:]
    elif num == 2:
        dict_mod = {3:"DWI"}
        list_mod = [0,0,0,1,0]
        if fill_zeros:
            X_modality = X[:, 700:1400]
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = X[:, 700:1400]
    elif num == 3:
        dict_mod = {3:"DWI", 4:"DWIC"}
        list_mod = [0,0,0,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 700:1400], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 700:1400], X[:, 1400:]), axis=1)
    elif num == 4:
        dict_mod = {2:"FLAIR"}
        list_mod = [0,0,1,0,0]
        if fill_zeros:
            X_modality = X[:, 500:700]
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = X[:, 500:700]
    elif num == 5:
        dict_mod = {2:"FLAIR", 4:"DWIC"}
        list_mod = [0,0,1,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 500:700], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 500:700], X[:, 1400:]), axis=1)
    elif num == 6:
        dict_mod = {2:"FLAIR", 3:"DWI"}
        list_mod = [0,0,1,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 500:700], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 500:700], X[:, 700:1400]), axis=1)
    elif num == 7:
        dict_mod = {2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [0,0,1,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 500:700], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 500:700], X[:, 700:]), axis=1)
    elif num == 8:
        dict_mod = {1:"T2"}
        list_mod = [0,1,0,0,0]
        if fill_zeros:
            X_modality = X[:, 300:500]
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = X[:, 300:500]
    elif num == 9:
        dict_mod = {1:"T2", 4:"DWIC"}
        list_mod = [0,1,0,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 1400:]), axis=1)
    elif num == 10:
        dict_mod = {1:"T2", 3:"DWI"}
        list_mod = [0,1,0,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 700:1400]), axis=1)
    elif num == 11:
        dict_mod = {1:"T2", 3:"DWI", 4:"DWIC"}
        list_mod = [0,1,0,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 700:]), axis=1)
    elif num == 12:
        dict_mod = {1:"T2", 2:"FLAIR"}
        list_mod = [0,1,1,0,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700]), axis=1)
    elif num == 13:
        dict_mod = {1:"T2", 2:"FLAIR", 4:"DWIC"}
        list_mod = [0,1,1,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1) 
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 1400:]), axis=1)
    elif num == 14:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI"}
        list_mod = [0,1,1,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1) 
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 700:1400]), axis=1)
    elif num == 15:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [0,1,1,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1) 
        else:
            X_modality = np.concatenate((X[:, 300:500], X[:, 500:700], X[:, 700:]), axis=1)
    elif num == 16:
        dict_mod = {0:"T1"}
        list_mod = [1,0,0,0,0]
        if fill_zeros:
            X_modality = X[:, :300]
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = X[:, :300]
    elif num == 17:
        dict_mod = {0:"T1", 4:"DWIC"}
        list_mod = [1,0,0,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 1400:]), axis=1)
    elif num == 18:
        dict_mod = {0:"T1", 3:"DWI"}
        list_mod = [1,0,0,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 700:1400]), axis=1)
    elif num == 19:
        dict_mod = {0:"T1", 3:"DWI", 4:"DWIC"}
        list_mod = [1,0,0,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 700:]), axis=1)
    elif num == 20:
        dict_mod = {0:"T1", 2:"FLAIR"}
        list_mod = [1,0,1,0,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700]), axis=1)
    elif num == 21:
        dict_mod = {0:"T1", 2:"FLAIR", 4:"DWIC"}
        list_mod = [1,0,1,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 1400:]), axis=1)
    elif num == 22:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI"}
        list_mod = [1,0,1,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))), axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 700:1400]), axis=1)
    elif num == 23:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [1,0,1,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 500:700], X[:, 700:]), axis=1)
    elif num == 24:
        dict_mod = {0:"T1", 1:"T2"}
        list_mod = [1,1,0,0,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500]), axis=1)
    elif num == 25:
        dict_mod = {0:"T1", 1:"T2", 4:"DWIC"}
        list_mod = [1,1,0,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 1400:]), axis=1)
    elif num == 26:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI"}
        list_mod = [1,1,0,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 700:1400]), axis=1)
    elif num == 27:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI", 4:"DWIC"}
        list_mod = [1,1,0,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 700:]), axis=1)
    elif num == 28:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR"}
        list_mod = [1,1,1,0,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700]), axis=1)
    elif num == 29:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 4:"DWIC"}
        list_mod = [1,1,1,0,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 1400:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:        
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 1400:]), axis=1)
    elif num == 30:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI"}
        list_mod = [1,1,1,1,0]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 700:1400]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:        
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 700:1400]), axis=1)
    elif num == 31:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
        list_mod = [1,1,1,1,1]
        if fill_zeros:
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 700:]), axis=1)
            X_modality = np.concatenate((X_modality, np.zeros((X.shape[0], (X.shape[1]-X_modality.shape[1])))),axis=1)
        else:        
            X_modality = np.concatenate((X[:, :300], X[:, 300:500], X[:, 500:700], X[:, 700:]), axis=1)
    else:
        raise ValueError(f"num should be betwen 1 and 31, got {num}")

    return X_modality, dict_mod # type: ignore

def load_train_data(root: str, node_num: str, j: int):

    train_path = os.path.join(root, 'Node_'+node_num, 'Aug_Train_Data', 'ALL_Patients')  
    x_file = f"X_train_aug"
    y_file = f"Y_train_aug"
    x_mat_name = "X_aug_train"
    y_mat_name = "Y_aug_train"  

    raw_path_x = os.path.join(train_path, f"{x_file}.mat")
    raw_path_y = os.path.join(train_path, f"{y_file}.mat")

    # Load the data from .mat files
    X_mat_l = loadmat(raw_path_x)
    X_mat = X_mat_l[x_mat_name]

    X_mat_modality, dict_mod = get_modality(j, X_mat) # get X for the modality

    Y_mat_l = loadmat(raw_path_y)
    Y_mat = Y_mat_l[y_mat_name]
    Y_mat = Y_mat.reshape(Y_mat.shape[1],)

    # Count and print the number of 0s and 1s
    num_zeros = np.sum(Y_mat == 0)
    num_ones = np.sum(Y_mat == 1)
    print(f"Train data for Node {node_num}, Modality {j}: Class 0 count = {num_zeros}, Class 1 count = {num_ones}")

    return X_mat_modality, Y_mat, dict_mod
    # return X_mat, Y_mat

def load_test_data(root: str, node_num: str, j: int):

    val_path = os.path.join(root, 'Node_'+node_num, 'Orig_Val_Data', 'ALL_Patients')  
    x_file = f"X_valid_orig"
    y_file = f"Y_valid_orig"
    x_mat_name = "X_orig_valid"
    y_mat_name = "Y_orig_valid"  

    raw_path_x = os.path.join(val_path, f"{x_file}.mat")
    raw_path_y = os.path.join(val_path, f"{y_file}.mat")

    # Load the data from .mat files
    X_mat_l = loadmat(raw_path_x)
    X_mat = X_mat_l[x_mat_name]

    X_mat_modality, dict_mod = get_modality(j, X_mat) # get X for the modality
    # X_mat_modality, dict_mod = get_modality(j, X_mat, fill_zeros=True) # get X for the modality

    Y_mat_l = loadmat(raw_path_y)
    Y_mat = Y_mat_l[y_mat_name]
    Y_mat = Y_mat.reshape(Y_mat.shape[1],)

    # Count and print the number of 0s and 1s
    num_zeros = np.sum(Y_mat == 0)
    num_ones = np.sum(Y_mat == 1)
    print(f"Test data for Node {node_num}, Modality {j}: Class 0 count = {num_zeros}, Class 1 count = {num_ones}")

    return X_mat_modality, Y_mat, dict_mod

# choose model
# model = 'MLP'
model = 'RF'
# model = 'XGB'
# model = 'SVM'

# Main loop to run the baseline models over all the nodes (for all 3 trials)

node_numbers_with_smote = get_list_of_node_nums()

for node_num in node_numbers_with_smote:
    
    print(f'Node num: {node_num}')

    num_trials = 3
    
    val_bal_acc_per_modality_list = []

    # Train and test ALL modality combinations for ALL trials
    for j in range(1,32): 
        # load the data for the given node and given modality combination
        X_train, Y_train, dict_mod = load_train_data(root, node_num, j)
        # X_train, Y_train = load_train_data(root, node_num, j)

        # print(f"X_train shape: {X_train.shape}")
        
        X_test, Y_test, dict_mod = load_test_data(root, node_num, j)

        # print(f"X_test shape: {X_test.shape}")
        # raise ValueError("Stop here")

        print(f"Modality Combination: {dict_mod}")

        # Define the model
        if model == 'MLP':
            clf = MLP(hidden_layer_sizes=(256,), learning_rate_init=0.01, random_state=None, max_iter=1000, early_stopping=False)
        elif model == 'RF':
            clf = RF(n_estimators=100, random_state=None,)
        elif model == 'SVM':
            clf = SVM(C=1.0, kernel='rbf', max_iter=-1, random_state=None)
        else:
            raise NotImplementedError("Unknown Model.")

        val_bal_acc_list = []
        # Run 5 trials for each node
        for i in range(num_trials):
            print(f'Training Trial {i+1} of Node number {node_num}')

            # Train the model
            clf.fit(X_train, Y_train)

            # Test the model
            print(f'Evaluating Trial {i+1} of Node number: {node_num}')
            y_true = Y_test
            y_pred = clf.predict(X_test)

            # Evaluate Trained Model with evaluation metrics
            bal_acc = calculate_metrics(y_pred, y_true)  
            # print(f"Balanced Accuracy: {bal_acc}")

            val_bal_acc_list.append(bal_acc) 

        val_bal_acc_per_modality_list.append(np.mean(val_bal_acc_list))

    # Create a DataFrame
    headers_val = ['Node_'+node_num]

    df_val = pd.DataFrame(val_bal_acc_per_modality_list, columns=headers_val)

    # Saving to Excel
    # path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/baselines_node_level/" 
    path = f"/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/magmsforEZprediction/sota_node_level/{model}_Results/" 
    save_path = os.path.join(path, "Node_"+node_num, "Eval_Results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename_val = "RF_results_val_ALL_modality_combination_test.xlsx"
    save_filepath_val = os.path.join(save_path, filename_val)

    df_val.to_excel(save_filepath_val, index=False, sheet_name='Sheet1')

# Combine all node results into one dataframe

# Define the paths of your Excel files
base_path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/baselines_node_level/" 

# For FULL modality Only
node_nums = get_list_of_node_nums()

file_paths_val = []

for node_num in node_nums:
    file_path_val = os.path.join(base_path, "Node_"+node_num+"_Results", "Eval_Results", "RF_results_val_ALL_modality_combination_test.xlsx") # For FULL modality Only
    file_paths_val.append(file_path_val)

# Initialize an empty DataFrame
combined_df_val = pd.DataFrame()

# Loop through the files and stack the rows
for path in file_paths_val:
    # Load the Excel file
    df = pd.read_excel(path)  

    # Stack the rows
    combined_df_val = pd.concat([combined_df_val, df], axis=1) # For ALL modality combinations

# Reset the index to avoid duplicate row indices
combined_df_val = combined_df_val.reset_index(drop=True)

# Save the combined DataFrame to a new Excel file
# combined_df_val.to_excel('RF_results_val_ALL_modality_combination_combined_Right_Hemis.xlsx', index=False)
combined_df_val.to_excel('RF_results_val_ALL_modality_combination_combined_Right_Hemis_test.xlsx', index=False)




