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

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.TRAIN, shuffle: bool = False, node_num: int = 1, device=torch.device("cuda:0")) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle, device=device)
        self.mode = mode
        self.root = root
        self.node_num = node_num
        # torch.multiprocessing.set_sharing_strategy('file_system')

        # initialize path
        if self.mode == EZMode.TRAIN:
            # self.path = os.path.join(self.root,'Train_NonEZvsEZ_ALL')
            # self.RI_file = f"Train_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            # self.Conn_file = f"Train_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            # self.label_file = f"Train_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            # self.RI_mat_name = "Augmented_RI"
            # self.Conn_mat_name = "Augmented_Conn"
            # self.label_mat_name = "Augmented_label"
            self.path = os.path.join(self.root, 'Train_NonEZvsEZ_ALL_aug')
            self.x_file = f"X_train_aug_node{self.node_num}.mat"
            self.y_file = f"Y_train_aug_node{self.node_num}.mat"
            self.x_mat_name = "X_aug_train"
            self.y_mat_name = "Y_aug_train"

        elif self.mode == EZMode.VALIDATE:
            # self.path = os.path.join(self.root,'Valid_NonEZvsEZ_ALL')
            # self.RI_file = f"Valid_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            # self.Conn_file = f"Valid_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            # self.label_file = f"Valid_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            # self.RI_mat_name = "ModelCohort_NonEZvsEZ_RI"
            # self.Conn_mat_name = "ModelCohort_NonEZvsEZ_Conn"
            # self.label_mat_name = "ModelCohort_NonEZvsEZ_label"
            self.path = os.path.join(self.root, 'Val_NonEZvsEZ_ALL_aug')
            self.x_file = f"X_val_aug_node{self.node_num}.mat"
            self.y_file = f"Y_val_aug_node{self.node_num}.mat"
            self.x_mat_name = "X_aug_val"
            self.y_mat_name = "Y_aug_val"

        elif self.mode == EZMode.TEST:
            # self.path = os.path.join(self.root,'ValidCohort_NonEZvsEZ_ALL')
            # self.RI_file = f"ValidCohort_NonEZvsEZ_RI_node{self.node_num}_ALL.mat"
            # self.Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{self.node_num}_ALL.mat"
            # self.label_file = f"ValidCohort_NonEZvsEZ_label_node{self.node_num}_ALL.mat"
            # self.RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
            # self.Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
            # self.label_mat_name = "ValidCohort_NonEZvsEZ_label"
            self.path = os.path.join(self.root, 'Test_NonEZvsEZ_ALL_aug')
            self.x_file = f"X_test_aug_node{self.node_num}.mat"
            self.y_file = f"Y_test_aug_node{self.node_num}.mat"
            self.x_mat_name = "X_aug_test"
            self.y_mat_name = "Y_aug_test"

        else:
            raise NotImplementedError("Select either train, validate or test mode.")

    @property
    def data(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # raw_path_RI = os.path.join(self.root,self.path,self.RI_file)
        # raw_path_Conn = os.path.join(self.root,self.path,self.Conn_file)
        # raw_path_label = os.path.join(self.root,self.path,self.label_file)

        # """Load the Relative Intensity (RI) Data Matrix from .mat files.""" 
        # X_mat_l = loadmat(raw_path_RI)
        # X_mat_RI = X_mat_l[self.RI_mat_name]
        # # print(f'RI Feature Matrix Shape: {X_mat_RI.shape}')

        # # check for NaN values and replace NaN values with 0
        # if (np.isnan(X_mat_RI).any()):  
        #     X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

        # """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
        # X_mat_lconn = loadmat(raw_path_Conn)
        # X_mat_DWIC = X_mat_lconn[self.Conn_mat_name]  # DWIC matrix: 1x499
        # # print(f'DWIC Connectome profile matrix shape: {X_mat_DWIC.shape}')
                  
        # # check for NaN values and replace NaN values with 0
        # if (np.isnan(X_mat_DWIC).any()):
        #     X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

        # # X_combined = [X_mat_T1, X_mat_T2, X_mat_FLAIR, X_mat_DWI, X_mat_DWIC]
        # X_combined = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features
        # # print(f'X_combined matrix shape: {X_combined.shape}')

        # X_multi_modal: torch.Tensor = torch.from_numpy(X_combined)

        # """Load the Label Matrix from .mat files.""" 
        # Y_mat_l = loadmat(raw_path_label)
        # Y_mat_aug = Y_mat_l[self.label_mat_name]
        # # print(f'GT-Labels shape:{Y_mat_aug.shape}')
        # Y_mat_aug = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
        # Y_label: torch.Tensor = torch.from_numpy(Y_mat_aug) # for CrossEntropyLoss

        raw_path_x = os.path.join(self.path,self.x_file)
        raw_path_y = os.path.join(self.path,self.y_file)

        # Load the data from .mat files
        X_mat_l = loadmat(raw_path_x)
        X_mat = X_mat_l[self.x_mat_name]

        Y_mat_l = loadmat(raw_path_y)
        Y_mat = Y_mat_l[self.y_mat_name]
        Y_mat = Y_mat.reshape(Y_mat.shape[1],)

        X_multi_modal = X_mat
        Y_label = Y_mat

        return X_multi_modal, Y_label

    @property
    def unbatched_len(self) -> int:
        r"""Returns the total length of the dataset (before forming into batches)."""
        """Load the Label Matrix from .mat files.""" 
        # raw_path_label = os.path.join(self.root,self.path,self.label_file) 
        # Y_mat_l = loadmat(raw_path_label)
        # Y_mat_aug = Y_mat_l[self.label_mat_name]
        # Y_mat_aug = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
        # Y_label: torch.Tensor = torch.from_numpy(Y_mat_aug) # for CrossEntropyLoss

        raw_path_y = os.path.join(self.path,self.y_file)
        Y_mat_l = loadmat(raw_path_y)
        Y_mat = Y_mat_l[self.y_mat_name]
        # Y_label: torch.Tensor = torch.from_numpy(Y_mat) # for CrossEntropyLoss
        # return len(Y_mat)
        return Y_mat.shape[1]

    def __getitem__(self, index: Any) -> Any:
        r"""Gets the data object at index.
        """
        # Load the 1D vectors (images) and binary labels
        X_multi_modal, Y_label = self.data
        X_multi_modal: torch.Tensor = torch.from_numpy(X_multi_modal) 
        Y_label: torch.Tensor = torch.from_numpy(Y_label) # for CrossEntropyLoss
        X_multi_modal = X_multi_modal.float()
        return X_multi_modal[index], Y_label[index]

    @staticmethod
    def unpack_data(data: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
        # fetch input and label
        x, y = Dataset.unpack_data(data)
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), "Data should be valid `torch.Tensor`."

        # unpack data (b, 1899) -> [(b, m, f), ...]
        x_t1 = x[:, :300].unsqueeze(dim=1) # t1 only
        x_t2 = x[:, 300:500].unsqueeze(dim=1) # t2 only
        x_flair = x[:, 500:700].unsqueeze(dim=1) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        # x_all = x[:, :1400].unsqueeze(dim=1) # all modalities
        x_dwic = x[:, 1400:].unsqueeze(dim=1) # dwic only
        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


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
            self.path = os.path.join(self.root, 'Train_NonEZvsEZ_whole_brain_aug_separate_fold'+fold_no)
            self.x_file = f"X_train_aug_whole_brain_node"
            self.y_file = f"Y_train_aug_whole_brain_node"
            self.x_mat_name = "X_aug_train_node"
            self.y_mat_name = "Y_aug_train_node"
            # self.size = 81612 # fold 1
            # self.size = 81672 # fold 2
            # self.size = 83006 # fold 3
            # self.size = 82766 # fold 4
            # self.size = 84960 # fold 5

        elif self.mode == EZMode.VALIDATE:
            self.path = os.path.join(self.root, 'Val_NonEZvsEZ_whole_brain_aug_separate_fold'+fold_no)
            self.x_file = f"X_val_aug_whole_brain_node"
            self.y_file = f"Y_val_aug_whole_brain_node"
            self.x_mat_name = "X_aug_valid_node"
            self.y_mat_name = "Y_aug_valid_node"
            # self.size = 21892 # fold 1
            # self.size = 21832 # fold 2
            # self.size = 20498 # fold 3
            # self.size = 20738 # fold 4
            # self.size = 18544 # fold 5

        elif self.mode == EZMode.TEST:
            # For augmented test set
            self.path = os.path.join(self.root, 'Test_NonEZvsEZ_whole_brain_aug_separate')
            self.x_file = f"X_test_aug_whole_brain_node"
            self.y_file = f"Y_test_aug_whole_brain_node"
            self.x_mat_name = "X_aug_test_node"
            self.y_mat_name = "Y_aug_test_node"

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
        x_t1 = x[:, :300].unsqueeze(dim=1) # t1 only
        x_t2 = x[:, 300:500].unsqueeze(dim=1) # t2 only
        x_flair = x[:, 500:700].unsqueeze(dim=1) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        x_dwic = x[:, 1400:].unsqueeze(dim=1) # dwic only

        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


class DatasetEZ_NodeLevel(Dataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        mode (EZMode): The training/validation/testing mode to load the data
        For each node number, there are several patients with either EZ (class 1) or non-EZ (class 0)
    """
    size: int
    root: str
    mode: EZMode

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.TRAIN, shuffle: bool = False, device=torch.device("cuda:0")) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle, device=device)
        self.mode = mode
        self.root = root

        # initialize path
        if self.mode == EZMode.TRAIN:
            self.path = os.path.join(self.root, 'Train_NonEZvsEZ_node_948_aug_separate')
            self.x_file = f"X_train_aug_node948_patient"
            self.y_file = f"Y_train_aug_node948_patient"
            self.x_mat_name = "X_aug_train_node948_patient"
            self.y_mat_name = "Y_aug_train_node948_patient"
            self.size = 64

        elif self.mode == EZMode.VALIDATE:
            self.path = os.path.join(self.root, 'Val_NonEZvsEZ_node_948_aug_separate')
            self.x_file = f"X_val_aug_node948_patient"
            self.y_file = f"Y_val_aug_node948_patient"
            self.x_mat_name = "X_aug_valid_node948_patient"
            self.y_mat_name = "Y_aug_valid_node948_patient"
            self.size = 40

        elif self.mode == EZMode.TEST:
            # For augmented test set
            self.path = os.path.join(self.root, 'Test_NonEZvsEZ_node_948_aug_separate')
            self.x_file = f"X_test_aug_node948_patient"
            self.y_file = f"Y_test_aug_node948_patient"
            self.x_mat_name = "X_aug_test_node948_patient"
            self.y_mat_name = "Y_aug_test_node948_patient"
            self.size = 20

        else:
            raise NotImplementedError("Select either train, validate or test mode.")

    @property
    def unbatched_len(self) -> int:
        r"""Returns the total length of the dataset (before forming into batches)."""
        """Load the Label Matrix from .mat files.""" 

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
        x_t1 = x[:, :300].unsqueeze(dim=1) # t1 only
        x_t2 = x[:, 300:500].unsqueeze(dim=1) # t2 only
        x_flair = x[:, 500:700].unsqueeze(dim=1) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        x_dwic = x[:, 1400:].unsqueeze(dim=1) # dwic only

        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y
    
class DatasetEZ_WB_Control(Dataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
    """
    root: str
    mode: EZMode

    def __init__(self, batch_size: int, root: str, drop_last: bool = False, mode: EZMode = EZMode.VALIDATE, shuffle: bool = False, device=torch.device("cuda:0")) -> None:
        super().__init__(batch_size, drop_last=drop_last, shuffle=shuffle, device=device)
        self.mode = mode
        self.root = root

        # initialize path
        if self.mode == EZMode.TRAIN:
            raise NotImplementedError("Train data not saved for control patients.") 

        elif self.mode == EZMode.VALIDATE:
            # with z-score norm
            self.path = os.path.join(self.root, 'Val_NonEZvsEZ_whole_brain_control_separate')
            # NO z-score norm
            # self.path = os.path.join(self.root, 'Val_NonEZvsEZ_whole_brain_control_separate_NO_Norm')
            self.x_file = f"X_val_control_whole_brain_node"
            self.y_file = f"Y_val_control_whole_brain_node"
            self.x_mat_name = "X_control_valid_node"
            self.y_mat_name = "Y_control_valid_node"

        else:
            raise NotImplementedError("Select either train or validate mode.")

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
        x_t1 = x[:, :300].unsqueeze(dim=1) # t1 only
        x_t2 = x[:, 300:500].unsqueeze(dim=1) # t2 only
        x_flair = x[:, 500:700].unsqueeze(dim=1) # flair only
        x_dwi = x[:, 700:1400].unsqueeze(dim=1) # dwi only
        x_dwic = x[:, 1400:].unsqueeze(dim=1) # dwic only

        return [x_t1, x_t2, x_flair, x_dwi, x_dwic], y


if __name__ == "__main__":

    # print("Whole Brain EZ Dataset ...")
    # ez_dataset = DatasetEZ_WB(batch_size=1, root='/home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/SMOTE_Augmented_Data/', drop_last=False, mode=EZMode.VALIDATE, fold_no="1", shuffle=True)

    # print(ez_dataset.unbatched_len)
    # # print((ez_dataset.__getitem__(0))[0][4].shape)
    # # print((ez_dataset.__getitem__(0))[1])

    # X_combined, Y_label = ez_dataset.__getitem__(0)
    # print(X_combined.shape)
    # print(Y_label.shape)

    # print("Whole Brain EZ Dataset ...")
    # ez_dataset = DatasetEZ_NodeLevel(batch_size=1, root='/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/baselines_all_hemispheres/', drop_last=False, mode=EZMode.TEST, shuffle=True)

    # print(ez_dataset.unbatched_len)
    # # print((ez_dataset.__getitem__(0))[0][4].shape)
    # # print((ez_dataset.__getitem__(0))[1])

    # X_combined, Y_label = ez_dataset.__getitem__(0)
    # print(X_combined.shape)
    # print(Y_label)

    print("Whole Brain Control Dataset ...")
    ez_dataset = DatasetEZ_WB_Control(batch_size=1, root='/home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/', drop_last=False, mode=EZMode.VALIDATE, shuffle=False)

    print(ez_dataset.unbatched_len)
    # print((ez_dataset.__getitem__(0))[0][4].shape)
    # print((ez_dataset.__getitem__(0))[1])

    X_combined, Y_label = ez_dataset.__getitem__(0)
    print(X_combined.shape)
    print(Y_label.shape)
