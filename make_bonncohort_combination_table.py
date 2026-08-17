# Build the modality-combination table of the Bonn cohort for reporting
#
# One row per sequence combination, in the same publication layout as the additional cohort:
#
#   #  Combo  T1w  T2w  FLAIR  DWI  DWIC  Left_Hemis_Mean_BA  Right_Hemis_Mean_BA
#   1      4    0    0      1    0     0                 ...                  ...
#   2     16    1    0      0    0     0                 ...                  ...
#   3     20    1    0      1    0     0                 ...                  ...
#
# The 0/1 indicators come from `get_target_dict` in eval_left.py, the same mapping the evaluation
# used, so row n of this table is exactly the combination evaluated as combination n. `Combo` keeps
# the original 1-31 number so the rows line up with the AddCohort table.
#
# Each hemisphere value is the mean over that hemisphere's nodes of the per-node balanced accuracy,
# where a node's balanced accuracy is the best of its 3 trials.
#
# The Bonn subjects have T1 and FLAIR only, so there are 3 combinations rather than 31: the model is
# reused unchanged from the AddCohort training and the absent sequences are dropped from its
# target_dict at inference.
#
# Usage:
#   python make_bonncohort_combination_table.py
import os
import pandas as pd

from eval_left import get_target_dict, get_modality_name
from eval_bonn_left import get_t1_flair_combinations


BASE_PATH = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/BonnCohort/'

# the sequence columns, in the order get_target_dict indexes them
SEQUENCE_COLUMNS = ["T1w", "T2w", "FLAIR", "DWI", "DWIC"]


def hemisphere_results(hemisphere: str, base_path: str = BASE_PATH) -> tuple[pd.DataFrame, pd.Series]:
    r"""
    Read the max-over-trials sheet of one hemisphere.

    Args:
        hemisphere (str): Either `left` or `right`.
        base_path (str): The root of the BonnCohort results tree.

    Returns: A `tuple` of the per-node balanced accuracies indexed by combination name, and the node
        number of each column.
    """
    path = os.path.join(base_path, f"BonnCohort_{hemisphere}_combined.xlsx")

    # max_over_trials already holds, for each node, the best of its 3 trials
    df = pd.read_excel(path, sheet_name="max_over_trials").set_index("Combination")

    nodes = pd.Series([int(c.split("_")[1]) for c in df.columns], index=df.columns)
    return df, nodes


def build_table(keep_ez_only: bool = False, base_path: str = BASE_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    r"""
    Args:
        keep_ez_only (bool): Whether to average only over the nodes whose validation split contains
            at least one EZ subject.
        base_path (str): The root of the BonnCohort results tree.

    Returns: A `tuple` of the combination table in the publication layout and the node count of each
        hemisphere.
    """
    # nodes whose Bonn validation split actually contains an EZ subject
    info = pd.read_excel(os.path.join(base_path, "Information", "info_bonn_cohort.xlsx"))
    has_ez = set(info.loc[info['EZ'] > 0, 'Node #'].astype(int))

    means = {}
    counts = {}

    for hemisphere in ["left", "right"]:
        df, nodes = hemisphere_results(hemisphere, base_path)

        columns = [c for c in df.columns if not keep_ez_only or nodes[c] in has_ez]
        means[hemisphere] = df[columns].mean(axis=1)
        counts[hemisphere] = len(columns)

    rows = []

    for index, num in enumerate(get_t1_flair_combinations(), start=1):
        dict_mod, list_mod = get_target_dict(num)
        name = get_modality_name(dict_mod)

        # the indicator vector and the target dict must describe the same combination
        assert [1 if i in dict_mod else 0 for i in range(5)] == list_mod, f"Combination {num} is inconsistent."

        # and this cohort can only have contributed T1 and FLAIR
        assert list_mod[1] == list_mod[3] == list_mod[4] == 0, f"Combination {num} uses a sequence the Bonn cohort does not have."

        row = {'#': index, 'Combo': num}
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

    save_filepath = os.path.join(base_path, "BonnCohort_combination_table.xlsx")

    with pd.ExcelWriter(save_filepath) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, index=False, sheet_name=name)

    print(f"\nSaved to {save_filepath}")
    return save_filepath


if __name__ == "__main__":
    main()

    print("\nDone!")
