# Load the dataset with different modalities for EZ prediction
import os
import numpy as np
from scipy.io import loadmat

import torch
from torchmanager_core.typing import Any, Enum, Tuple, Sequence
from torchmanager.data import Dataset


class EZMode(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


class DatasetEZ(Dataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        mode (EZMode): The training/validation/testing mode to load the data
        node_num (string): The node number of the brain for which we perform the training/validation and testing
        For each node number, there are several patients with either EZ (class 1) or non-EZ (class 0)
    """    
    root: str
    mode: EZMode
    node_num: int

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.TRAIN, shuffle: bool = False, node_num: int = 1) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle)
        self.mode = mode
        self.root = root
        self.node_num = node_num
        # torch.multiprocessing.set_sharing_strategy('file_system')

        # initialize path
        if self.mode == EZMode.TRAIN:
            self.path = os.path.join(self.root,'Train_NonEZvsEZ_ALL')
            self.RI_file = f"Train_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            self.Conn_file = f"Train_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            self.label_file = f"Train_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            self.RI_mat_name = "Augmented_RI"
            self.Conn_mat_name = "Augmented_Conn"
            self.label_mat_name = "Augmented_label"

        elif self.mode == EZMode.VALIDATE:
            self.path = os.path.join(self.root,'Valid_NonEZvsEZ_ALL')
            self.RI_file = f"Valid_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            self.Conn_file = f"Valid_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            self.label_file = f"Valid_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            self.RI_mat_name = "ModelCohort_NonEZvsEZ_RI"
            self.Conn_mat_name = "ModelCohort_NonEZvsEZ_Conn"
            self.label_mat_name = "ModelCohort_NonEZvsEZ_label"

        elif self.mode == EZMode.TEST:
            self.path = os.path.join(self.root,'ValidCohort_NonEZvsEZ_ALL')
            self.RI_file = f"ValidCohort_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            self.Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            self.label_file = f"ValidCohort_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            self.RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
            self.Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
            self.label_mat_name = "ValidCohort_NonEZvsEZ_label"

        else:
            raise NotImplementedError("Select either train, validate or test mode.")

    @property
    def data(self) -> Tuple[torch.Tensor, torch.Tensor]:

        raw_path_RI = os.path.join(self.root,self.path,self.RI_file)
        raw_path_Conn = os.path.join(self.root,self.path,self.Conn_file)
        raw_path_label = os.path.join(self.root,self.path,self.label_file)

        """Load the Relative Intensity (RI) Data Matrix from .mat files.""" 
        X_mat_l = loadmat(raw_path_RI)
        X_mat_RI = X_mat_l[self.RI_mat_name]
        # print(f'RI Feature Matrix Shape: {X_mat_RI.shape}')

        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_RI).any()):  
            X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

        """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
        X_mat_lconn = loadmat(raw_path_Conn)
        X_mat_DWIC = X_mat_lconn[self.Conn_mat_name]  # DWIC matrix: 1x499
        # print(f'DWIC Connectome profile matrix shape: {X_mat_DWIC.shape}')
                  
        # check for NaN values and replace NaN values with 0
        if (np.isnan(X_mat_DWIC).any()):
            X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

        # X_combined = [X_mat_T1, X_mat_T2, X_mat_FLAIR, X_mat_DWI, X_mat_DWIC]
        X_combined = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features
        # print(f'X_combined matrix shape: {X_combined.shape}')

        X_multi_modal: torch.Tensor = torch.from_numpy(X_combined)

        """Load the Label Matrix from .mat files.""" 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[self.label_mat_name]
        # print(f'GT-Labels shape:{Y_mat_aug.shape}')
        Y_mat_aug = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
        Y_label: torch.Tensor = torch.from_numpy(Y_mat_aug).long() # for CrossEntropyLoss

        return X_multi_modal, Y_label

    @property
    def unbatched_len(self) -> int:
        r"""Returns the total length of the dataset (before forming into batches)."""
        """Load the Label Matrix from .mat files.""" 
        raw_path_label = os.path.join(self.root,self.path,self.label_file) 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[self.label_mat_name]
        Y_mat_aug = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
        Y_label: torch.Tensor = torch.from_numpy(Y_mat_aug).long() # for CrossEntropyLoss
        return len(Y_label)

    def __getitem__(self, index: Any) -> Any:
        r"""Gets the data object at index.
        """
        # Load the 1D vectors (images) and binary labels
        X_multi_modal, Y_label = self.data

        return X_multi_modal[index], Y_label[index]

    @staticmethod
    def unpack_data(data: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
        # fetch input and label
        x, y = super().unpack_data(data)

        # unpack data (b, 1, 1899) -> [(b, 1, f), ...]
        x_t1 = x[:, :, :300]
        x_t2 = x[:, :, 300:500]
        x_flair = x[:, :, 500:700]
        x_dwi = x[:, :, 700:1400] 
        x_dwic = x[:, :, 1400:]
        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


if __name__ == "__main__":

    print("EZ Dataset ...")
    ez_dataset = DatasetEZ(batch_size=2, root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/', drop_last=False, mode=EZMode.TRAIN, shuffle=True, node_num=948)

    print(ez_dataset.unbatched_len)
    # print((ez_dataset.__getitem__(0))[0][4].shape)
    # print((ez_dataset.__getitem__(0))[1])

    X_combined, Y_label = ez_dataset.__getitem__(0)
    print(X_combined.shape)
    print(Y_label)
