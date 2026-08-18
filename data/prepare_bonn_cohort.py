# Stage the Bonn validation cohort into the per-patient layout read by DatasetEZ_Node
#
# The cohort is the public OpenNeuro dataset ds004199 (University of Bonn, CC0): 85 people with
# focal cortical dysplasia type II, imaged with T1 and FLAIR only. It covers exactly the same 711
# nodes as the additional cohort.
#
# Features come from the per-node export at BONN_EXPORT:
#   BonnCohort_NonEZvsEZ_RI_node{N}.mat     (85, 1400)  -> T1 | T2 | FLAIR | DWI
#   BonnCohort_NonEZvsEZ_Conn_node{N}.mat   (85, 499)   -> DWIC (connectome profile)
#   BonnCohort_NonEZvsEZ_label_node{N}.mat  (85, 1)     -> ALL ZEROS, see below
#
# Only T1 and FLAIR carry data. In the source, the T2 and DWI segments are NaN and the connectome is
# 100% NaN; the export ran nan_to_num over them, so those segments read back as zeros. They are
# missing acquisitions, not zero measurements, which is why evaluation drops them from the model's
# target_dict rather than feeding an all-zero branch.
#
# The labels had to be recovered. Every one of the 60,435 (85 x 711) entries in the export's label
# files is 0 - the export lost the label column. The labels live in the BIDS source as
# Bonn_Cohort_Label.mat, laid out subject-major over the 998-ROI Lausanne parcellation:
#
#     row = subject_index * 998 + (node_num - 1)
#
# That row index is the same one that reproduces the exported features bit-for-bit, and
# `verify_label_alignment` below proves features and labels share one subject ordering. The export
# itself is never modified: this script reads features from BOTH sources and asserts they agree.
import argparse
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import loadmat

try:  # `python data/prepare_bonn_cohort.py` from the repo root
    from prepare_add_cohort import MAX_LEFT_NODE, save_as_separate_patients
except ImportError:  # `from data.prepare_bonn_cohort import ...`
    from .prepare_add_cohort import MAX_LEFT_NODE, save_as_separate_patients


# The BIDS source that produced the export. Holds the labels the export dropped.
BONN_SOURCE = '/media/user1/Data/OpenSource_Data/Bonn_data/'

# The per-node export, used to verify the staged features are unchanged.
BONN_EXPORT = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/BonnData/'

NUM_SUBJECTS = 85

# ROIs per hemisphere, and in total, in the Lausanne parcellation the source is laid out over.
NUM_ROIS_PER_HEMI = 499
NUM_ROIS = 998

# Sub-directory name under Node_{N}/. The whole cohort shares one group: unlike the additional
# cohort there is no per-subject split, since every subject is missing the same three sequences.
DIR_BONN = "Bonn_Val_Data_85"

# The same 85 subjects with the labels exactly as the export ships them, i.e. all zeros. Every
# node-subject pair is therefore treated as non-EZ, and balanced accuracy collapses to the non-EZ
# accuracy (the true-negative rate for EZ detection) because BalancedAccuracyScore falls back to
# specificity = sensitivity when one class is absent. Staged so that evaluation can be run against
# the export as-is, alongside the recovered-label evaluation.
DIR_BONN_ASEXPORTED = "Bonn_Val_Data_85_asexported"

# Segments of the 1899-feature vector that must be empty for this cohort, per unpack_data.
ABSENT_SEGMENTS = {"T2": (300, 500), "DWI": (700, 1400), "DWIC": (1400, 1899)}


def load_source(source: str = BONN_SOURCE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Load the whole cohort from the BIDS source in one pass.

    The arrays are subject-major over `NUM_ROIS` ROIs, so a node is a strided row selection rather
    than a file read. Loading once beats reopening 2,133 files.

    Args:
        source (str): The BIDS source directory.

    Returns: A `tuple` of the (84830, 1400) RI matrix, the (84830, 499) connectome matrix and the
        (84830,) binary label vector.
    """
    RI = loadmat(os.path.join(source, "Bonn_Cohort_RI.mat"))["Bonn_Cohort_RI"]
    Conn = loadmat(os.path.join(source, "Bonn_Cohort_Conn.mat"))["Bonn_Cohort_Conn"]
    Label = loadmat(os.path.join(source, "Bonn_Cohort_Label.mat"))["Bonn_Cohort_Label"].reshape(-1)

    expected = NUM_SUBJECTS * NUM_ROIS
    assert RI.shape == (expected, 1400), f"Expected a ({expected}, 1400) RI matrix, got {RI.shape}."
    assert Conn.shape == (expected, NUM_ROIS_PER_HEMI), f"Expected a ({expected}, {NUM_ROIS_PER_HEMI}) Conn matrix, got {Conn.shape}."
    assert Label.shape == (expected,), f"Expected {expected} labels, got {Label.shape}."

    # the ROI index of every row must be the ascending 1..998 block this script assumes
    ROI = loadmat(os.path.join(source, "Bonn_Cohort_ROI.mat"))["Bonn_Cohort_ROI"].reshape(-1)
    assert np.array_equal(ROI.reshape(NUM_SUBJECTS, NUM_ROIS), np.tile(np.arange(1, NUM_ROIS + 1), (NUM_SUBJECTS, 1))), \
        "Bonn_Cohort_ROI is not a subject-major block of ascending ROI indices, so the row formula does not hold."

    return RI, Conn, Label


def verify_label_alignment(RI: np.ndarray, Label: np.ndarray, source: str = BONN_SOURCE) -> None:
    r"""
    Prove that `Bonn_Cohort_Label` is row-aligned with `Bonn_Cohort_RI`.

    This is the check the whole cohort rests on: the export carries no labels, so the labels are
    taken from a different file and nothing else guarantees the two share a subject ordering.

    `Structure_GM_RI.m` builds the source arrays side-dependently per subject - `BonnData_*` holds
    the lesion-side hemisphere and `BonnData_*_OPP` the other one. `BonnData_ROI` records which ROI
    indices each subject's block used, which recovers that side vector from the data alone, with no
    reliance on any metadata file. If the same side vector reassembles both the feature matrix and
    the label vector, they were written in the same subject order.

    Note that subject *identity* is deliberately not established here. The 85 rows are NOT ordered by
    participant_id (matching them that way agrees with participants.tsv on only 34/85, i.e. chance).
    Per-node balanced accuracy needs row alignment, which this proves, not identity.

    Args:
        RI (np.ndarray): The (84830, 1400) `Bonn_Cohort_RI` matrix.
        Label (np.ndarray): The (84830,) `Bonn_Cohort_Label` vector.
        source (str): The BIDS source directory.
    """
    def load(name: str, var: str) -> np.ndarray:
        return loadmat(os.path.join(source, name))[var]

    # per subject: is the lesion-side block the LEFT half of the parcellation?
    roi_ipsi = load("BonnData_ROI.mat", "BonnData_ROI").reshape(NUM_SUBJECTS, NUM_ROIS_PER_HEMI)
    is_left = roi_ipsi[:, 0] <= NUM_ROIS_PER_HEMI

    # the side vector must agree with participants.tsv in aggregate
    assert int(is_left.sum()) == 46 and int((~is_left).sum()) == 39, \
        f"Expected 46 left / 39 right lesion sides, got {int(is_left.sum())} / {int((~is_left).sum())}."

    def reassemble(ipsi: np.ndarray, contra: np.ndarray) -> np.ndarray:
        r"""Place each subject's ipsi/contra blocks into their anatomical halves."""
        out = np.empty((NUM_SUBJECTS, NUM_ROIS) + ipsi.shape[2:], dtype=ipsi.dtype)
        for i in range(NUM_SUBJECTS):
            first, second = (ipsi[i], contra[i]) if is_left[i] else (contra[i], ipsi[i])
            out[i, :NUM_ROIS_PER_HEMI], out[i, NUM_ROIS_PER_HEMI:] = first, second
        return out

    # labels first: small arrays, and the claim that actually matters
    lab_ipsi = load("BonnData_Label.mat", "BonnData_Label").reshape(NUM_SUBJECTS, NUM_ROIS_PER_HEMI)
    lab_contra = load("BonnData_Label_OPP.mat", "BonnData_Label").reshape(NUM_SUBJECTS, NUM_ROIS_PER_HEMI)
    assert np.array_equal(reassemble(lab_ipsi, lab_contra), Label.reshape(NUM_SUBJECTS, NUM_ROIS)), \
        "Bonn_Cohort_Label does not reassemble from BonnData_Label/_OPP under the BonnData_ROI side vector."

    # then features, under the *same* side vector. Loaded one at a time to bound peak memory.
    cohort = np.nan_to_num(RI.reshape(NUM_SUBJECTS, NUM_ROIS, -1), nan=0.0)
    feat_ipsi = np.nan_to_num(load("Norm_BonnData_RI.mat", "Norm_BonnData_RI").reshape(NUM_SUBJECTS, NUM_ROIS_PER_HEMI, -1), nan=0.0)
    feat_contra = np.nan_to_num(load("Norm_BonnData_RI_OPP.mat", "Norm_BonnData_RI").reshape(NUM_SUBJECTS, NUM_ROIS_PER_HEMI, -1), nan=0.0)
    assert np.array_equal(reassemble(feat_ipsi, feat_contra), cohort), \
        "Bonn_Cohort_RI does not reassemble from Norm_BonnData_RI/_OPP under the BonnData_ROI side vector."

    print("Label alignment verified: features and labels share one subject ordering and side assignment.")


def node_rows(node_num: str) -> np.ndarray:
    r"""The 85 source row indices holding one node, one per subject."""
    return np.arange(NUM_SUBJECTS) * NUM_ROIS + (int(node_num) - 1)


def exported_labels(node_num: str, export: str = BONN_EXPORT) -> np.ndarray:
    r"""
    The labels of one node exactly as the export ships them.

    These are all zero for every node in the cohort; the assertion is what makes that explicit
    rather than assumed, so if the export is ever regenerated with real labels this staging picks
    them up instead of silently writing zeros.

    Args:
        node_num (str): The node number.
        export (str): The per-node export directory.

    Returns: The (85,) label vector in `np.ndarray`.
    """
    Y = loadmat(os.path.join(export, f"BonnCohort_NonEZvsEZ_label_node{node_num}.mat"))["BonnCohort_NonEZvsEZ_label"]
    Y = Y.reshape(-1).astype(int)

    assert Y.shape == (NUM_SUBJECTS,), f"Node {node_num}: expected {NUM_SUBJECTS} exported labels, got {Y.shape}."

    unexpected = set(np.unique(Y).tolist()) - {0, 1}
    assert not unexpected, f"Node {node_num}: exported labels must be binary, found {sorted(unexpected)}."

    return Y


def load_node(RI: np.ndarray, Conn: np.ndarray, Label: np.ndarray, node_num: str) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Assemble one node from the pre-loaded source arrays.

    Args:
        RI (np.ndarray): The (84830, 1400) RI matrix.
        Conn (np.ndarray): The (84830, 499) connectome matrix.
        Label (np.ndarray): The (84830,) label vector.
        node_num (str): The node number.

    Returns: A `tuple` of the (85, 1899) feature matrix and the (85,) binary label vector.
    """
    rows = node_rows(node_num)

    # the absent sequences are NaN at source; zero them exactly as the export did
    X_RI = np.nan_to_num(RI[rows], nan=0.0)
    X_DWIC = np.nan_to_num(Conn[rows], nan=0.0)

    X_combined = np.concatenate((X_RI, X_DWIC), axis=1)  # using both RI and Conn features
    Y = Label[rows].astype(int)

    return X_combined, Y


def check_node(node_num: str, X: np.ndarray, Y: np.ndarray, export: str = BONN_EXPORT) -> None:
    r"""Verify the assumptions this script relies on, and that the features match the export."""
    assert X.shape == (NUM_SUBJECTS, 1899), f"Node {node_num}: expected a ({NUM_SUBJECTS}, 1899) feature matrix, got {X.shape}."
    assert Y.shape == (NUM_SUBJECTS,), f"Node {node_num}: expected {NUM_SUBJECTS} labels, got {Y.shape}."

    unexpected = set(np.unique(Y).tolist()) - {0, 1}
    assert not unexpected, f"Node {node_num}: labels must be binary, found {sorted(unexpected)}."

    # this cohort has T1 and FLAIR only; the other three segments must be empty
    for name, (lo, hi) in ABSENT_SEGMENTS.items():
        assert not X[:, lo:hi].any(), f"Node {node_num}: expected {name} to be absent, found non-zero values."

    # and T1/FLAIR must not be empty, which would mean a failed acquisition rather than a missing one
    assert X[:, 0:300].any(axis=1).all(), f"Node {node_num}: some subjects have an all-zero T1 segment."
    assert X[:, 500:700].any(axis=1).all(), f"Node {node_num}: some subjects have an all-zero FLAIR segment."

    # the staged features must be bit-for-bit the ones already exported, so that recovering the
    # labels from a second file cannot silently change the inputs
    exported_RI = loadmat(os.path.join(export, f"BonnCohort_NonEZvsEZ_RI_node{node_num}.mat"))["BonnCohort_NonEZvsEZ_RI"]
    exported_Conn = loadmat(os.path.join(export, f"BonnCohort_NonEZvsEZ_Conn_node{node_num}.mat"))["BonnCohort_NonEZvsEZ_Conn"]

    assert np.array_equal(X[:, :1400], exported_RI), f"Node {node_num}: staged RI differs from the exported RI."
    assert np.array_equal(X[:, 1400:], exported_Conn), f"Node {node_num}: staged Conn differs from the exported Conn."


def get_list_of_node_nums(root: str = BONN_EXPORT) -> list[str]:
    r"""Return every node number present in the Bonn cohort export, sorted numerically."""
    prefix, suffix = "BonnCohort_NonEZvsEZ_label_node", ".mat"

    node_nums = [f[len(prefix):-len(suffix)] for f in os.listdir(root) if f.startswith(prefix) and f.endswith(suffix)]

    return sorted(node_nums, key=int)


def main(source: str, export: str, save_path_left: str, save_path_right: str, save_path_info: str, check_alignment: bool = True) -> None:

    node_nums = get_list_of_node_nums(export)

    print(f"Total number of nodes in the Bonn cohort: {len(node_nums)}")

    RI, Conn, Label = load_source(source)

    if check_alignment:
        verify_label_alignment(RI, Label, source)

    # the source ships its own node list; it must be the one that was exported
    listed = pd.read_excel(os.path.join(source, "Lausanne_node_indices.xlsx"))["Node #"].astype(int).tolist()
    assert [int(i) for i in node_nums] == sorted(listed), "Lausanne_node_indices.xlsx does not match the exported node list."

    node_numbers = []
    hemispheres = []

    num_nonEZs = []
    num_EZs = []
    num_exported_EZs = []

    for i in node_nums:
        print(f"Loading BonnCohort for Node num: {i}")

        X_node, Y_node = load_node(RI, Conn, Label, node_num=i)
        check_node(i, X_node, Y_node, export)

        is_left = int(i) <= MAX_LEFT_NODE
        save_path = save_path_left if is_left else save_path_right

        print('Y_bonn: %s' % Counter(Y_node))

        save_dir = os.path.join(save_path, 'Node_' + i, DIR_BONN)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_as_separate_patients(save_dir, X_node, Y_node)

        # the same features with the export's own labels, so the cohort can also be scored exactly
        # as it ships. Identical X, only Y differs.
        Y_exported = exported_labels(i, export)

        save_dir_exported = os.path.join(save_path, 'Node_' + i, DIR_BONN_ASEXPORTED)
        if not os.path.exists(save_dir_exported):
            os.makedirs(save_dir_exported)
        save_as_separate_patients(save_dir_exported, X_node, Y_exported)

        num_exported_EZs.append(np.sum(Y_exported))

        # record the class balance for this node
        node_numbers.append(i)
        hemispheres.append("left" if is_left else "right")

        num_nonEZs.append(len(Y_node) - np.sum(Y_node))
        num_EZs.append(np.sum(Y_node))

    # dictionary of lists
    info_dict = {'Node #': node_numbers, 'Hemisphere': hemispheres, 'NonEZ': num_nonEZs, 'EZ': num_EZs,
                 'EZ_as_exported': num_exported_EZs}

    df = pd.DataFrame(info_dict)

    # saving the dataframe
    if not os.path.exists(save_path_info):
        os.makedirs(save_path_info)

    filename = "info_bonn_cohort.xlsx"
    save_filepath = os.path.join(save_path_info, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    ez = np.array(num_EZs)
    left = np.array(hemispheres) == "left"

    print(f"\nNodes staged: {len(node_numbers)} ({hemispheres.count('left')} left, {hemispheres.count('right')} right)")
    print(f"EZ labels recovered: {int(ez.sum())}")
    print(f"Nodes with at least one EZ subject: {int((ez > 0).sum())} "
          f"(left {int((ez[left] > 0).sum())}, right {int((ez[~left] > 0).sum())})")
    print(f"EZ labels in the export itself: {int(np.sum(num_exported_EZs))} "
          f"(staged separately under {DIR_BONN_ASEXPORTED})")
    print(f"Information file saved at: {save_filepath}")

    ################################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage the Bonn validation cohort for DatasetEZ_Node.")
    parser.add_argument("--source", type=str, default=BONN_SOURCE, help="The BIDS source directory holding Bonn_Cohort_*.mat.")
    parser.add_argument("--export", type=str, default=BONN_EXPORT, help="The per-node export, used to verify the staged features.")
    parser.add_argument("--skip_alignment_check", action="store_true", help="Skip the feature/label alignment proof (it loads ~1.5GB).")
    args = parser.parse_args()

    # The staged validation data lives next to Orig_Val_Data inside the existing node folders, so that
    # training and validation still share a single --data_dir argument.
    save_path_left = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/'
    save_path_right = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/'

    # Results and summaries for this cohort are kept under their own BonnCohort tree
    save_path_info = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/BonnCohort/Information/'

    main(args.source, args.export, save_path_left, save_path_right, save_path_info, check_alignment=not args.skip_alignment_check)

    print("\nDone!")
