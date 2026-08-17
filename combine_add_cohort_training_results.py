# Combine the per-node training-time results of the additional cohort into a single Excel file
#
# train_left.py / train_right.py write one results_val.xlsx and one results_train.xlsx per node,
# each holding the balanced accuracy, sensitivity and specificity of the 3 trials. The validation
# numbers are measured on the 17 subjects of the additional cohort that have all five sequences,
# using all 5 sequences at inference.
#
# The per-node EZ counts are merged in so that nodes whose validation split contains no EZ subject
# can be filtered out: for those, BalancedAccuracyScore falls back to specificity = sensitivity, so
# the balanced accuracy is really just the non-EZ accuracy.
#
# Usage:
#   python combine_add_cohort_training_results.py
import os
import pandas as pd

from data.prepare_add_cohort import get_list_of_node_nums, MAX_LEFT_NODE


BASE_PATH = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/AddCohort/'
ADD_DATASET_ROOT = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/AddDataSet/'

BAL_ACC_COLUMNS = ['Val_Balanced_Accuracy_1', 'Val_Balanced_Accuracy_2', 'Val_Balanced_Accuracy_3']


def collect(mode: str, base_path: str = BASE_PATH, add_dataset_root: str = ADD_DATASET_ROOT) -> pd.DataFrame:
    r"""
    Read the per-node result files of one mode.

    Args:
        mode (str): Either `val` or `train`.
        base_path (str): The root of the AddCohort results tree.
        add_dataset_root (str): The flat directory of the additional cohort, used to list the nodes.

    Returns: A `pd.DataFrame` with one row per node.
    """
    rows = []
    missing = []

    for node_num in get_list_of_node_nums(add_dataset_root):
        hemisphere = "left" if int(node_num) <= MAX_LEFT_NODE else "right"
        hemis_dir = "Left_Hemis" if hemisphere == "left" else "Right_Hemis"

        file_path = os.path.join(base_path, hemis_dir, "Node_" + node_num, "Results", f"results_{mode}.xlsx")

        if not os.path.exists(file_path):
            missing.append(node_num)
            continue

        df = pd.read_excel(file_path)
        df.insert(1, 'Hemisphere', hemisphere)
        rows.append(df)

    if missing:
        print(f"WARNING: no results_{mode}.xlsx for {len(missing)} node(s): {missing[:10]}")

    return pd.concat(rows, axis=0).reset_index(drop=True)


def main(base_path: str = BASE_PATH, add_dataset_root: str = ADD_DATASET_ROOT) -> str:
    df_val = collect("val", base_path, add_dataset_root)
    df_train = collect("train", base_path, add_dataset_root)

    # merge the class balance of each node's validation split
    info_path = os.path.join(base_path, "Information", "info_add_cohort.xlsx")
    if os.path.exists(info_path):
        info = pd.read_excel(info_path)[['Node #', 'NonEZ-withDWIC', 'EZ-withDWIC']]
        df_val = df_val.merge(info, on='Node #', how='left')

        # a node with no EZ subject cannot produce a meaningful balanced accuracy
        df_val.insert(2, 'Has_EZ_subject', df_val['EZ-withDWIC'] > 0)

    # the best and the average of the 3 trials, for convenience
    df_val['Max_Val_Bal_Acc'] = df_val[BAL_ACC_COLUMNS].max(axis=1)
    df_val['Mean_Val_Bal_Acc'] = df_val[BAL_ACC_COLUMNS].mean(axis=1)

    save_filepath = os.path.join(base_path, "AddCohort_training_results_all_nodes.xlsx")

    with pd.ExcelWriter(save_filepath) as writer:
        df_val.to_excel(writer, index=False, sheet_name='validation')
        df_train.to_excel(writer, index=False, sheet_name='training')

    print(f"validation: {len(df_val)} nodes, training: {len(df_train)} nodes")

    if 'Has_EZ_subject' in df_val:
        informative = df_val[df_val['Has_EZ_subject']]
        print(f"nodes with at least one EZ subject in the validation split: {len(informative)}")
        print(f"  mean over trials, those nodes only: max={informative['Max_Val_Bal_Acc'].mean():.4f}, mean={informative['Mean_Val_Bal_Acc'].mean():.4f}")
        print(f"  mean over trials, all nodes:        max={df_val['Max_Val_Bal_Acc'].mean():.4f}, mean={df_val['Mean_Val_Bal_Acc'].mean():.4f}")

    print(f"Saved to {save_filepath}")
    return save_filepath


if __name__ == "__main__":
    main()

    print("\nDone!")
