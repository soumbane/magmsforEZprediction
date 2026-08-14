# Combine the per-node evaluation results of the additional cohort into a single Excel file
#
# eval_left.py / eval_right.py write one file per node per subject group, with one row per modality
# combination and one column per training trial. This script concatenates those columns across nodes
# and adds the aggregations over trials.
#
# Examples:
#   python combine_add_cohort_results.py --hemisphere left --group 17
#   python combine_add_cohort_results.py --hemisphere right --group 13
import argparse
import os
import pandas as pd

from data.prepare_add_cohort import get_list_of_node_nums, MAX_LEFT_NODE


BASE_PATH = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/AddCohort/'

# the flat directory of the additional cohort, used as the authoritative list of node numbers
ADD_DATASET_ROOT = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/AddDataSet/'

# per-node filename written by eval_left.py / eval_right.py for each subject group
GROUP_FILES = {
    "17": "results_add17_ALL31_trials.xlsx",  # 17 subjects with all five sequences, 31 combinations
    "13": "results_add13_DWICfree15_trials.xlsx",  # 13 subjects without DWIC, 15 combinations
}


def get_node_nums(hemisphere: str, add_dataset_root: str = ADD_DATASET_ROOT) -> list[str]:
    r"""
    The node numbers of one hemisphere, taken from the additional cohort itself rather than from a
    hand-maintained list.
    """
    node_nums = get_list_of_node_nums(add_dataset_root)

    if hemisphere == "left":
        return [n for n in node_nums if int(n) <= MAX_LEFT_NODE]
    else:
        return [n for n in node_nums if int(n) > MAX_LEFT_NODE]


def combine(hemisphere: str, group: str, base_path: str = BASE_PATH, skip_missing: bool = False, add_dataset_root: str = ADD_DATASET_ROOT) -> str:
    r"""
    Combine the per-node result files of one hemisphere and one subject group.

    Args:
        hemisphere (str): Either `left` or `right`.
        group (str): Either `17` (all five sequences) or `13` (no DWIC).
        base_path (str): The root of the AddCohort results tree.
        skip_missing (bool): Whether to skip nodes whose result file has not been written yet.
        add_dataset_root (str): The flat directory of the additional cohort, used to list the nodes.

    Returns: The path in `str` of the combined Excel file.
    """
    node_nums = get_node_nums(hemisphere, add_dataset_root)
    filename = GROUP_FILES[group]
    hemis_dir = "Left_Hemis" if hemisphere == "left" else "Right_Hemis"

    combinations = None
    per_trial_columns = {}
    missing = []

    for node_num in node_nums:
        file_path = os.path.join(base_path, hemis_dir, "Node_" + node_num + "_Results", "Eval_Results", filename)

        if not os.path.exists(file_path):
            missing.append(node_num)
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing results for node {node_num}: {file_path}. Pass --skip_missing to ignore.")

        df = pd.read_excel(file_path)

        # the modality combination of each row must line up across every node
        if combinations is None:
            combinations = df['Combination'].tolist()
        else:
            assert df['Combination'].tolist() == combinations, f"Node {node_num} has a different set of modality combinations."

        for column in df.columns:
            if column == 'Combination':
                continue
            per_trial_columns[f"Node_{node_num}_{column}"] = df[column].tolist()

    assert combinations is not None, "No result files were found."

    if missing:
        print(f"WARNING: skipped {len(missing)} node(s) with no results: {missing}")

    # one column per (node, trial)
    df_per_trial = pd.DataFrame({'Combination': combinations, **per_trial_columns})

    # aggregate the trials of each node into a single column
    trial_columns = [c for c in df_per_trial.columns if c != 'Combination']
    nodes_present = sorted({c.rsplit('_Trial_', 1)[0] for c in trial_columns}, key=lambda n: int(n.split('_')[1]))

    max_columns = {}
    mean_columns = {}

    for node in nodes_present:
        columns = [c for c in trial_columns if c.rsplit('_Trial_', 1)[0] == node]
        max_columns[node] = df_per_trial[columns].max(axis=1)
        mean_columns[node] = df_per_trial[columns].mean(axis=1)

    df_max = pd.DataFrame({'Combination': combinations, **max_columns})
    df_mean = pd.DataFrame({'Combination': combinations, **mean_columns})

    save_filepath = os.path.join(base_path, f"AddCohort_{hemisphere}_group{group}_combined.xlsx")

    with pd.ExcelWriter(save_filepath) as writer:
        df_per_trial.to_excel(writer, index=False, sheet_name='per_trial')
        df_max.to_excel(writer, index=False, sheet_name='max_over_trials')
        df_mean.to_excel(writer, index=False, sheet_name='mean_over_trials')

    print(f"{hemisphere} hemisphere, group {group}: {len(combinations)} combinations x {len(nodes_present)} nodes ({len(trial_columns)} trial columns)")
    print(f"Saved to {save_filepath}")

    return save_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine the additional-cohort evaluation results across nodes.")
    parser.add_argument("--hemisphere", type=str, default="left", choices=["left", "right"], help="The hemisphere to combine, default is `left`.")
    parser.add_argument("--group", type=str, default="17", choices=["17", "13"], help="The subject group to combine: `17` with all five sequences or `13` without DWIC, default is `17`.")
    parser.add_argument("--base_path", type=str, default=BASE_PATH, help="The root of the AddCohort results tree.")
    parser.add_argument("--add_dataset_root", type=str, default=ADD_DATASET_ROOT, help="The flat directory of the additional cohort, used to list the nodes.")
    parser.add_argument("--skip_missing", action="store_true", default=False, help="The flag to skip nodes that have no results yet.")
    args = parser.parse_args()

    combine(args.hemisphere, args.group, base_path=args.base_path, skip_missing=args.skip_missing, add_dataset_root=args.add_dataset_root)

    print("\nDone!")
