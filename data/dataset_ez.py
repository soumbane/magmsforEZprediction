# Load the dataset with different modalities for EZ prediction
import os
import numpy as np
from scipy.io import loadmat  # type: ignore

import torch
from torch.nn import functional as F
from torchmanager_core.typing import Any, Enum, Tuple
from torchmanager.data import Dataset


class EZMode(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


class DatasetEZ_WB(Dataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        mode (EZMode): The training/validation/testing mode to load the data
        For each node number, there are several patients with either EZ (class 1) or non-EZ (class 0)
    """
    size: int
    root: str
    mode: EZMode

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.TRAIN, fold_no: str = "1", shuffle: bool = False, device=torch.device("cuda:0")) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle, device=device)
        self.mode = mode
        self.root = root

        # initialize path
        if self.mode == EZMode.TRAIN:
            self.path = os.path.join(self.root, 'SMOTE_Augmented_Data','Train_NonEZvsEZ_WB_smoteaug')

            self.x_file = f"X_train_aug_WB_node"
            self.y_file = f"Y_train_aug_WB_node"
            self.x_mat_name = "X_aug_train_node"
            self.y_mat_name = "Y_aug_train_node"
            
        elif self.mode == EZMode.VALIDATE:
            self.path = os.path.join(self.root, 'Original_Patient_Data', 'Test_NonEZvsEZ_WB_orig')
            
            self.x_file = f"X_test_orig_WB_node"
            self.y_file = f"Y_test_orig_WB_node"
            self.x_mat_name = "X_orig_test_node"
            self.y_mat_name = "Y_orig_test_node"
            
        elif self.mode == EZMode.TEST:
            self.path = os.path.join(self.root, 'Original_Patient_Data', 'ValidationCohort_NonEZvsEZ_WB_orig')
            
            self.x_file = f"X_valid_orig_WB_node"
            self.y_file = f"Y_valid_orig_WB_node"
            self.x_mat_name = "X_orig_valid_node"
            self.y_mat_name = "Y_orig_valid_node"

        else:
            raise NotImplementedError("Select either train, validate or test mode.")

    @property
    def unbatched_len(self) -> int:
        r"""Returns the total length of the dataset (before forming into batches)."""
        """Load the Label Matrix from .mat files."""

        self.size = (len(os.listdir(self.path)))//2
        
        return self.size

    def __getitem__(self, index: Any) -> Any:
        r"""Gets the data object at index.
        """
        raw_path_x = os.path.join(self.path, f"{self.x_file}{index}.mat")
        raw_path_y = os.path.join(self.path, f"{self.y_file}{index}.mat")

        # Load the data from .mat files
        X_mat_l = loadmat(raw_path_x)
        X_mat = X_mat_l[self.x_mat_name + str(index)]

        Y_mat_l = loadmat(raw_path_y)
        Y_mat = Y_mat_l[self.y_mat_name + str(index)]
        Y_mat = Y_mat.reshape(Y_mat.shape[1],)

        X_multi_modal = X_mat
        Y_label = Y_mat

        # Load the 1D vectors (images) and binary labels
        X_multi_modal: torch.Tensor = torch.from_numpy(X_multi_modal) 
        Y_label: torch.Tensor = torch.from_numpy(Y_label) # for CrossEntropyLoss
        Y_label = Y_label.squeeze(dim=0)
        X_multi_modal = X_multi_modal.float()
        X_multi_modal = X_multi_modal.squeeze(dim=0)
    
        return X_multi_modal, Y_label

    @staticmethod
    def unpack_data(data: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
        # fetch input and label
        x, y = Dataset.unpack_data(data)
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "Data should be valid `torch.Tensor`."

        # unpack data (b, 1899) -> [(b, m, f), ...] to include the modality dimension
        x_t1 = F.pad(x[:, :300].unsqueeze(dim=1), (200,200)) # t1 only
        x_t2 = F.pad(x[:, 300:500].unsqueeze(dim=1), (250,250)) # t2 only
        x_flair = F.pad(x[:, 500:700].unsqueeze(dim=1), (250,250)) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        x_dwic = F.pad(x[:, 1400:].unsqueeze(dim=1), (100,101)) # dwic only

        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


if __name__ == "__main__":    

    print("Lobe-level EZ Dataset ...")
    ez_dataset = DatasetEZ_WB(batch_size=1, root='/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Lobe_Data/', drop_last=False, mode=EZMode.TRAIN, shuffle=False)

    print(ez_dataset.unbatched_len)
    # print((ez_dataset.__getitem__(0))[0][4].shape)
    # print((ez_dataset.__getitem__(0))[1])

    X_combined, Y_label = ez_dataset[0]
    print(X_combined.shape)
    print(Y_label)
