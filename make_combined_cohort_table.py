# Build the joint AddCohort + BonnCohort combination table for the paper
#
# One row per sequence combination, 1-31, with both external cohorts side by side:
#
#   #  T1w  T2w  FLAIR  DWI  DWIC  AddCohort_Left  AddCohort_Right  BonnCohort_Left  BonnCohort_Right
#   1    0    0      0    0     1          0.6797           0.6889               NA                NA
#   ...
#   4    0    0      1    0     0          0.6769           0.6698           0.7280            0.6483
#   ...
#  31    1    1      1    1     1          0.8702           0.8624               NA                NA
#
# The AddCohort subjects have all five sequences, so all 31 rows are populated. The Bonn subjects
# (OpenNeuro ds004199) were scanned with T1 and FLAIR only - there is no T2, DWI or DWIC acquisition
# at all - so only the 3 non-empty subsets of {T1, FLAIR} exist for them:
#
#     row 4 = FLAIR, row 16 = T1, row 20 = T1-FLAIR
#
# The other 28 rows are reported as NA rather than as numbers. They could be *computed* by handing
# the model an all-zero tensor in place of each missing sequence, but that does not measure the
# sequence: a zero input still produces a non-zero constant from the encoder (conv bias plus
# normalisation), and since the branches are summed at inference those constants shift the decision.
# Measured on node 58, adding the three empty branches to T1-FLAIR flipped 15 of 85 predictions.
# So a number in those cells would be an artefact of zero-filling, not evidence about T2/DWI/DWIC.
#
# Each value is the mean over that hemisphere's nodes of the per-node balanced accuracy, where a
# node's balanced accuracy is the best of its 3 trials. The 0/1 indicators come from
# `get_target_dict`, the same mapping both evaluations used.
#
# Usage:
#   python make_combined_cohort_table.py
import os
import pandas as pd

from eval_left import get_target_dict, get_modality_name
from eval_bonn_left import get_t1_flair_combinations

import make_addcohort_combination_table as addcohort
import make_bonncohort_combination_table as bonncohort


# the sequence columns, in the order get_target_dict indexes them
SEQUENCE_COLUMNS = ["T1w", "T2w", "FLAIR", "DWI", "DWIC"]

# what to write where a cohort has no such acquisition
NOT_AVAILABLE = "NA"

SAVE_PATH = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/'


def cohort_means(keep_ez_only: bool) -> tuple[dict[str, dict[str, pd.Series]], dict[str, dict[str, int]]]:
    r"""
    The per-combination hemisphere means of both cohorts.

    Args:
        keep_ez_only (bool): Whether to average only over the nodes whose validation split contains
            at least one EZ subject. The two cohorts have different such node sets, so each is
            filtered against its own composition file.

    Returns: A `tuple` of the means keyed by cohort then hemisphere, and the matching node counts.
    """
    # each cohort records its EZ counts in its own information file, under its own column name
    add_info = pd.read_excel(os.path.join(addcohort.BASE_PATH, "Information", "info_add_cohort.xlsx"))
    add_ez = set(add_info.loc[add_info['EZ-withDWIC'] > 0, 'Node #'].astype(int))

    bonn_info = pd.read_excel(os.path.join(bonncohort.BASE_PATH, "Information", "info_bonn_cohort.xlsx"))
    bonn_ez = set(bonn_info.loc[bonn_info['EZ'] > 0, 'Node #'].astype(int))

    sources = {
        "AddCohort": (addcohort.hemisphere_results, add_ez),
        "BonnCohort": (bonncohort.hemisphere_results, bonn_ez),
    }

    means: dict[str, dict[str, pd.Series]] = {}
    counts: dict[str, dict[str, int]] = {}

    for cohort, (read, has_ez) in sources.items():
        means[cohort] = {}
        counts[cohort] = {}

        for hemisphere in ["left", "right"]:
            df, nodes = read(hemisphere)

            columns = [c for c in df.columns if not keep_ez_only or nodes[c] in has_ez]
            means[cohort][hemisphere] = df[columns].mean(axis=1)
            counts[cohort][hemisphere] = len(columns)

    return means, counts


def build_table(keep_ez_only: bool = False) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    r"""
    Args:
        keep_ez_only (bool): Whether to average only over the nodes that contain an EZ subject.

    Returns: A `tuple` of the 31-row joint table and the node counts of each cohort and hemisphere.
    """
    means, counts = cohort_means(keep_ez_only)

    # the only combinations the Bonn cohort has the acquisitions for
    bonn_available = set(get_t1_flair_combinations())

    rows = []

    for num in range(1, 32):
        dict_mod, list_mod = get_target_dict(num)
        name = get_modality_name(dict_mod)

        # the indicator vector and the target dict must describe the same combination
        assert [1 if i in dict_mod else 0 for i in range(5)] == list_mod, f"Combination {num} is inconsistent."

        row = {'#': num}
        row.update(dict(zip(SEQUENCE_COLUMNS, list_mod)))

        row['AddCohort_Left'] = means['AddCohort']['left'][name]
        row['AddCohort_Right'] = means['AddCohort']['right'][name]

        if num in bonn_available:
            row['BonnCohort_Left'] = means['BonnCohort']['left'][name]
            row['BonnCohort_Right'] = means['BonnCohort']['right'][name]
        else:
            # the Bonn subjects have no T2/DWI/DWIC acquisition, so this combination does not exist
            row['BonnCohort_Left'] = NOT_AVAILABLE
            row['BonnCohort_Right'] = NOT_AVAILABLE

        rows.append(row)

    table = pd.DataFrame(rows)

    # the Bonn columns must be NA everywhere except the three combinations it can support
    populated = set(table.loc[table['BonnCohort_Left'] != NOT_AVAILABLE, '#'])
    assert populated == bonn_available, f"Bonn values appear at {sorted(populated)}, expected {sorted(bonn_available)}."

    return table, counts


def build_specificity_table() -> tuple[pd.DataFrame, dict[str, int]]:
    r"""
    The Bonn cohort with every label forced non-EZ: a specificity analysis.

    With no positives of class 1, `BalancedAccuracyScore` falls back to
    `specificity = sensitivity`, so the number is the **non-EZ accuracy** - the fraction of the 85
    subjects the model correctly declines to call EZ, i.e. the true-negative rate for EZ detection.
    It is not a balanced accuracy and must not be put in the same column as one.

    There is no `nodes_with_EZ` counterpart: by construction no node contains an EZ subject here.

    Note this is a deliberately constructed label set, not the contents of any file. It was first
    run when the `/BonnData/` export shipped all-zero labels, so the two coincided at the time; the
    export has since been regenerated with real labels and no longer does.

    Returns: A `tuple` of the 31-row table and the node count of each hemisphere.
    """
    means = {}
    counts = {}

    for hemisphere in ["left", "right"]:
        path = os.path.join(bonncohort.BASE_PATH, f"BonnCohort_{hemisphere}_all_nonEZ_combined.xlsx")
        df = pd.read_excel(path, sheet_name="max_over_trials").set_index("Combination")

        means[hemisphere] = df.mean(axis=1)
        counts[hemisphere] = len(df.columns)

    available = set(get_t1_flair_combinations())

    rows = []

    for num in range(1, 32):
        dict_mod, list_mod = get_target_dict(num)
        name = get_modality_name(dict_mod)

        row = {'#': num}
        row.update(dict(zip(SEQUENCE_COLUMNS, list_mod)))

        if num in available:
            row['BonnCohort_NonEZ_Acc_Left'] = means['left'][name]
            row['BonnCohort_NonEZ_Acc_Right'] = means['right'][name]
        else:
            row['BonnCohort_NonEZ_Acc_Left'] = NOT_AVAILABLE
            row['BonnCohort_NonEZ_Acc_Right'] = NOT_AVAILABLE

        rows.append(row)

    return pd.DataFrame(rows), counts


def main(save_path: str = SAVE_PATH) -> str:
    sheets = {}

    for sheet_name, keep_ez_only in [("all_nodes", False), ("nodes_with_EZ", True)]:
        table, counts = build_table(keep_ez_only)
        sheets[sheet_name] = table

        print(f"\n=== {sheet_name} ===")
        print(f"  AddCohort  nodes: left {counts['AddCohort']['left']}, right {counts['AddCohort']['right']}")
        print(f"  BonnCohort nodes: left {counts['BonnCohort']['left']}, right {counts['BonnCohort']['right']}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # the Bonn specificity analysis, if that run exists
    specificity = os.path.join(bonncohort.BASE_PATH, "BonnCohort_left_all_nonEZ_combined.xlsx")

    if os.path.exists(specificity):
        table, counts = build_specificity_table()
        sheets['bonn_specificity'] = table

        print(f"\n=== bonn_specificity: NON-EZ ACCURACY, not balanced accuracy ===")
        print(f"  BonnCohort nodes: left {counts['left']}, right {counts['right']} (all 711, no EZ split possible)")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print(f"\nSkipping the specificity sheet: {specificity} not found.")

    save_filepath = os.path.join(save_path, "Combined_AddCohort_BonnCohort_table.xlsx")

    with pd.ExcelWriter(save_filepath) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, index=False, sheet_name=name)

    print(f"\nBonn columns are populated only at rows {sorted(get_t1_flair_combinations())} "
          f"(FLAIR, T1, T1-FLAIR); that cohort has no T2/DWI/DWIC acquisition.")
    print(f"Saved to {save_filepath}")
    return save_filepath


if __name__ == "__main__":
    main()

    print("\nDone!")
