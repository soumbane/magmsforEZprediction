# Stage the additional validation cohort (AddDataSet) into the per-patient layout read by DatasetEZ_Node
#
# The AddDataSet is stored flat, one set of files per node:
#   AddCohort_NonEZvsEZ_RI_node{N}.mat     (30, 1400)  -> T1 | T2 | FLAIR | DWI
#   AddCohort_NonEZvsEZ_Conn_node{N}.mat   (30, 499)   -> DWIC (connectome profile)
#   AddCohort_NonEZvsEZ_label_node{N}.mat  (30, 1)     -> binary labels, 1 = EZ
#
# 13 of the 30 subjects have no DWIC acquisition; their Conn rows are all zeros in every node. Those
# subjects are written to a separate folder so they can be evaluated with DWIC dropped from the
# model's target_dict, rather than being fed an all-zero DWIC branch.
#
# Unlike the original cohort, the labels here are already binary: there is no IZ code to drop and no
# 2|3 -> 1 remapping to apply.
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import loadmat, savemat


# Subjects without any DWIC data, 0-indexed (1-indexed: 1,3,5,10,11,13,14,15,26,27,28,29,30).
# These rows are all-zero in Conn for every one of the 711 nodes.
NO_DWIC_SUBJECTS = [0, 2, 4, 9, 10, 12, 13, 14, 25, 26, 27, 28, 29]

# The remaining 17 subjects, which have all five sequences.
WITH_DWIC_SUBJECTS = [i for i in range(30) if i not in NO_DWIC_SUBJECTS]

# Sub-directory names under Node_{N}/. Named after DWIC availability rather than group size, since
# the cohort also happens to split 17/13 by SOZ laterality and that is a different partition.
DIR_WITH_DWIC = "Add_Val_Data_17_withDWIC"
DIR_NO_DWIC = "Add_Val_Data_13_noDWIC"

# Nodes numbered at or below this belong to the left hemisphere.
MAX_LEFT_NODE = 479


def load_node(root: str, node_num: str):
    r"""
    Load one node of the additional cohort.

    Args:
        root (str): The directory holding the flat AddCohort .mat files.
        node_num (str): The node number.

    Returns: a `tuple` of the (30, 1899) feature matrix and the (30,) binary label vector.
    """
    raw_path_RI = os.path.join(root, f"AddCohort_NonEZvsEZ_RI_node{node_num}.mat")
    raw_path_Conn = os.path.join(root, f"AddCohort_NonEZvsEZ_Conn_node{node_num}.mat")
    raw_path_label = os.path.join(root, f"AddCohort_NonEZvsEZ_label_node{node_num}.mat")

    """Load the Relative Intensity (RI) Data Matrix from .mat files."""
    X_mat_RI = loadmat(raw_path_RI)["AddCohort_NonEZvsEZ_RI"]  # RI matrix: 30x1400

    # check for NaN values and replace NaN values with 0
    if np.isnan(X_mat_RI).any():
        X_mat_RI = np.nan_to_num(X_mat_RI, nan=0)

    """Load the Connectome Profile (DWIC) Matrix from .mat files."""
    X_mat_DWIC = loadmat(raw_path_Conn)["AddCohort_NonEZvsEZ_Conn"]  # DWIC matrix: 30x499

    # check for NaN values and replace NaN values with 0
    if np.isnan(X_mat_DWIC).any():
        X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

    X_combined = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1)  # using both RI and Conn features

    """Load the Label Matrix from .mat files."""
    Y_mat = loadmat(raw_path_label)["AddCohort_NonEZvsEZ_label"]
    Y_mat = Y_mat.reshape(Y_mat.shape[0],).astype(int)

    return X_combined, Y_mat, X_mat_DWIC


def check_node(node_num: str, X: np.ndarray, Y: np.ndarray, X_DWIC: np.ndarray) -> None:
    r"""Verify the assumptions this script relies on for a single node."""
    assert X.shape == (30, 1899), f"Node {node_num}: expected a (30, 1899) feature matrix, got {X.shape}."
    assert Y.shape == (30,), f"Node {node_num}: expected 30 labels, got {Y.shape}."

    unexpected = set(np.unique(Y).tolist()) - {0, 1}
    assert not unexpected, f"Node {node_num}: labels must be binary, found {sorted(unexpected)}."

    # every subject without DWIC must have an all-zero connectome profile at every node
    missing = [i for i in NO_DWIC_SUBJECTS if X_DWIC[i].any()]
    assert not missing, f"Node {node_num}: subjects {[i + 1 for i in missing]} were expected to have no DWIC."


def save_as_separate_patients(save_dir: str, X: np.ndarray, Y: np.ndarray) -> None:
    r"""
    Write one .mat per patient, using the same names as the original validation data so that
    `DatasetEZ_Node` can read it without special-casing.
    """
    for i in range(len(Y)):
        savemat(os.path.join(save_dir, 'X_valid_orig_patient' + str(i) + '.mat'), {"X_orig_valid_patient" + str(i): X[i, :]})
        savemat(os.path.join(save_dir, 'Y_valid_orig_patient' + str(i) + '.mat'), {"Y_orig_valid_patient" + str(i): Y[i]})


def get_list_of_node_nums(root: str) -> list[str]:
    r"""Return every node number present in the additional cohort, sorted numerically."""
    node_nums = []

    for f in os.listdir(root):
        if f.startswith("AddCohort_NonEZvsEZ_label_node") and f.endswith(".mat"):
            node_nums.append(f[len("AddCohort_NonEZvsEZ_label_node"):-len(".mat")])

    return sorted(node_nums, key=int)


def main(root: str, save_path_left: str, save_path_right: str, save_path_info: str) -> None:

    node_nums = get_list_of_node_nums(root)

    print(f"Total number of nodes in the additional cohort: {len(node_nums)}")

    node_numbers = []
    hemispheres = []

    num_nonEZs_with_dwic = []
    num_EZs_with_dwic = []

    num_nonEZs_no_dwic = []
    num_EZs_no_dwic = []

    for i in node_nums:
        print(f"Loading AddCohort for Node num: {i}")

        X_node, Y_node, X_DWIC = load_node(root, node_num=i)
        check_node(i, X_node, Y_node, X_DWIC)

        is_left = int(i) <= MAX_LEFT_NODE
        save_path = save_path_left if is_left else save_path_right

        # split the cohort by DWIC availability
        X_with_dwic, Y_with_dwic = X_node[WITH_DWIC_SUBJECTS, :], Y_node[WITH_DWIC_SUBJECTS]
        X_no_dwic, Y_no_dwic = X_node[NO_DWIC_SUBJECTS, :], Y_node[NO_DWIC_SUBJECTS]

        print('Y_add_with_DWIC: %s' % Counter(Y_with_dwic))
        print('Y_add_no_DWIC: %s' % Counter(Y_no_dwic))

        # save the 17 subjects that have all five sequences
        save_dir_with_dwic = os.path.join(save_path, 'Node_' + i, DIR_WITH_DWIC)
        if not os.path.exists(save_dir_with_dwic):
            os.makedirs(save_dir_with_dwic)
        save_as_separate_patients(save_dir_with_dwic, X_with_dwic, Y_with_dwic)

        # save the 13 subjects without DWIC
        save_dir_no_dwic = os.path.join(save_path, 'Node_' + i, DIR_NO_DWIC)
        if not os.path.exists(save_dir_no_dwic):
            os.makedirs(save_dir_no_dwic)
        save_as_separate_patients(save_dir_no_dwic, X_no_dwic, Y_no_dwic)

        # record the class balance of both groups for this node
        node_numbers.append(i)
        hemispheres.append("left" if is_left else "right")

        num_nonEZs_with_dwic.append(len(Y_with_dwic) - np.sum(Y_with_dwic))
        num_EZs_with_dwic.append(np.sum(Y_with_dwic))

        num_nonEZs_no_dwic.append(len(Y_no_dwic) - np.sum(Y_no_dwic))
        num_EZs_no_dwic.append(np.sum(Y_no_dwic))

    # dictionary of lists
    info_dict = {'Node #': node_numbers, 'Hemisphere': hemispheres, 'NonEZ-withDWIC': num_nonEZs_with_dwic, 'EZ-withDWIC': num_EZs_with_dwic, 'NonEZ-noDWIC': num_nonEZs_no_dwic, 'EZ-noDWIC': num_EZs_no_dwic}

    df = pd.DataFrame(info_dict)

    # saving the dataframe
    if not os.path.exists(save_path_info):
        os.makedirs(save_path_info)

    filename = "info_add_cohort.xlsx"
    save_filepath = os.path.join(save_path_info, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    print(f"\nNodes staged: {len(node_numbers)} ({hemispheres.count('left')} left, {hemispheres.count('right')} right)")
    print(f"Nodes with at least one EZ subject - withDWIC: {int(np.sum(np.array(num_EZs_with_dwic) > 0))}, noDWIC: {int(np.sum(np.array(num_EZs_no_dwic) > 0))}")
    print(f"Information file saved at: {save_filepath}")

    ################################################################################################################


if __name__ == "__main__":

    # Root folder holding the flat AddCohort .mat files
    root = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/AddDataSet/'

    # The staged validation data lives next to Orig_Val_Data inside the existing node folders, so that
    # training and validation still share a single --data_dir argument.
    save_path_left = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/'
    save_path_right = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/'

    # Results and summaries for this cohort are kept under their own AddCohort tree
    save_path_info = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/AddCohort/Information/'

    main(root, save_path_left, save_path_right, save_path_info)
