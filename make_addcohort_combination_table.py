# Build the modality-combination table of the additional cohort for reporting
#
# One row per sequence combination, in the publication layout:
#
#   #  T1w  T2w  FLAIR  DWI  DWIC  Left_Hemis_Mean_BA  Right_Hemis_Mean_BA
#   1    0    0      0    0     1              0.6797               0.6889
#   2    0    0      0    1     0              0.7042               0.6868
#   ...
#  31    1    1      1    1     1              0.8702               0.8624
#
# The 0/1 indicators come from `get_target_dict` in eval_left.py, the same mapping the evaluation
# used, so row n of this table is exactly the combination evaluated as combination n.
#
# Each hemisphere value is the mean over that hemisphere's nodes of the per-node balanced accuracy,
# where a node's balanced accuracy is the best of its 3 training trials.
#
# This covers the 17 subjects that have all five sequences: the model is trained once with all five
# and then evaluated on each of the 31 non-empty sequence combinations.
#
# Usage:
#   python make_addcohort_combination_table.py
import os
import pandas as pd

from eval_left import get_target_dict, get_modality_name


BASE_PATH = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/AddCohort/'

# the sequence columns, in the order get_target_dict indexes them
SEQUENCE_COLUMNS = ["T1w", "T2w", "FLAIR", "DWI", "DWIC"]


def hemisphere_results(hemisphere: str, base_path: str = BASE_PATH) -> tuple[pd.DataFrame, pd.Series]:
    r"""
    Read the max-over-trials sheet of one hemisphere.

    Args:
        hemisphere (str): Either `left` or `right`.
        base_path (str): The root of the AddCohort results tree.

    Returns: A `tuple` of the per-node balanced accuracies indexed by combination name, and the node
        number of each column.
    """
    path = os.path.join(base_path, f"AddCohort_{hemisphere}_group17_combined.xlsx")

    # max_over_trials already holds, for each node, the best of its 3 trials
    df = pd.read_excel(path, sheet_name="max_over_trials").set_index("Combination")

    nodes = pd.Series([int(c.split("_")[1]) for c in df.columns], index=df.columns)
    return df, nodes


def build_table(keep_ez_only: bool = False, base_path: str = BASE_PATH) -> pd.DataFrame:
    r"""
    Args:
        keep_ez_only (bool): Whether to average only over the nodes whose validation split contains
            at least one EZ subject.
        base_path (str): The root of the AddCohort results tree.

    Returns: A `pd.DataFrame` of the 31 combinations in the publication layout.
    """
    # nodes whose 17-subject validation split actually contains an EZ subject
    info = pd.read_excel(os.path.join(base_path, "Information", "info_add_cohort.xlsx"))
    has_ez = set(info.loc[info['EZ-withDWIC'] > 0, 'Node #'].astype(int))

    means = {}
    counts = {}

    for hemisphere in ["left", "right"]:
        df, nodes = hemisphere_results(hemisphere, base_path)

        columns = [c for c in df.columns if not keep_ez_only or nodes[c] in has_ez]
        means[hemisphere] = df[columns].mean(axis=1)
        counts[hemisphere] = len(columns)

    rows = []

    for num in range(1, 32):
        dict_mod, list_mod = get_target_dict(num)
        name = get_modality_name(dict_mod)

        # the indicator vector and the target dict must describe the same combination
        assert [1 if i in dict_mod else 0 for i in range(5)] == list_mod, f"Combination {num} is inconsistent."

        row = {'#': num}
        row.update(dict(zip(SEQUENCE_COLUMNS, list_mod)))
        row['Left_Hemis_Mean_BA'] = means['left'][name]
        row['Right_Hemis_Mean_BA'] = means['right'][name]
        rows.append(row)

    return pd.DataFrame(rows), counts


def main(base_path: str = BASE_PATH) -> str:
    sheets = {}

    for sheet_name, keep_ez_only in [("all_nodes", False), ("nodes_with_EZ", True)]:
        table, counts = build_table(keep_ez_only, base_path)
        sheets[sheet_name] = table

        print(f"\n=== {sheet_name}: left n={counts['left']} nodes, right n={counts['right']} nodes ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    save_filepath = os.path.join(base_path, "AddCohort_group17_combination_table.xlsx")

    with pd.ExcelWriter(save_filepath) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, index=False, sheet_name=name)

    print(f"\nSaved to {save_filepath}")
    return save_filepath


if __name__ == "__main__":
    main()

    print("\nDone!")
