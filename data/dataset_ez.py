# Load the dataset with different modalities for EZ prediction for each node number
import os
import numpy as np
from scipy.io import loadmat  # type: ignore

import torch
from torch.nn import functional as F
from torchmanager_core.typing import Any, Enum, Tuple
from torchmanager.data import Dataset


class EZMode(Enum):
    TRAIN = "train"
    TEST = "test"


class DatasetEZ_Node(Dataset):
    r"""
    Args:
        root (string): Root directory where the dataset is saved.
        mode (EZMode): The training/testing mode to load the data
        For each node number, there are several patients with either EZ (class 1) or non-EZ (class 0)
    """
    size: int
    root: str
    mode: EZMode

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.TRAIN, shuffle: bool = False, node_num: str = "1", device=torch.device("cuda:0")) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle, device=device)
        self.mode = mode
        self.root = root

        # initialize path
        if self.mode == EZMode.TRAIN:
            self.path = os.path.join(self.root, 'Node_'+node_num, 'Aug_Train_Data')

            self.x_file = f"X_train_aug_patient"
            self.y_file = f"Y_train_aug_patient"
            self.x_mat_name = "X_aug_train_patient"
            self.y_mat_name = "Y_aug_train_patient"
            
        elif self.mode == EZMode.TEST:
            self.path = os.path.join(self.root, 'Node_'+node_num, 'Orig_Val_Data') # ALL Val Patients
            # self.path = os.path.join(self.root, 'Node_'+node_num, 'MR0') # MR0 Val Patients
            # self.path = os.path.join(self.root, 'Node_'+node_num, 'MR1') # MR1 Val Patients
            # self.path = os.path.join(self.root, 'Node_'+node_num, 'SR') # SR Val Patients
            # self.path = os.path.join(self.root, 'Node_'+node_num, 'SF') # SF Val Patients
            
            self.x_file = f"X_valid_orig_patient"
            self.y_file = f"Y_valid_orig_patient"
            self.x_mat_name = "X_orig_valid_patient"
            self.y_mat_name = "Y_orig_valid_patient"

        else:
            raise NotImplementedError("Select either train or test mode.")

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

        # unpack data (b, 1899) -> [(b, m, f), ...] to include the modality dimension and then pad zeros to have same tensor shape
        x_t1 = F.pad(x[:, :300].unsqueeze(dim=1), (200,200)) # t1 only
        x_t2 = F.pad(x[:, 300:500].unsqueeze(dim=1), (250,250)) # t2 only
        x_flair = F.pad(x[:, 500:700].unsqueeze(dim=1), (250,250)) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        x_dwic = F.pad(x[:, 1400:].unsqueeze(dim=1), (100,101)) # dwic only

        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


if __name__ == "__main__":    

    print("Node-level EZ Dataset ...")
    ez_dataset = DatasetEZ_Node(batch_size=1, root='/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/', drop_last=False, mode=EZMode.TEST, shuffle=False, node_num="888")

    print(ez_dataset.unbatched_len)
    # print((ez_dataset.__getitem__(0))[0][4].shape)
    # print((ez_dataset.__getitem__(0))[1])

    X_combined, Y_label = ez_dataset[0]
    print(X_combined.shape)
    print(Y_label)
