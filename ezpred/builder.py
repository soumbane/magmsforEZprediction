import torch
from magnet import MAGNET2
from .networks import Basic

from .nn import MSFE, SHFE, SCH
from .nn.fusion import Fusion, FusionType


# def build(num_classes: int = 2, /, out_main_ch: int = 64, main_downsample: bool = True, *, out_filters: int = 128, filters_t1: list[int] = [32,64,128], filters_t2: list[int] = [32,64,128], filters_flair: list[int] = [32,64,128], filters_dwi: list[int] = [32,64,128], filters_dwic: list[int] = [32,64,128], filters_shfe: list[int] = [128,128], fusion: FusionType = FusionType.MID_MEAN, train_modality: str = "ALL") -> MAGNET2[MSFE, Fusion, torch.nn.Sequential]:
def build(num_classes: int = 2, /, out_main_ch: int = 64, main_downsample: bool = True, *, out_filters: int = 128, filters_t1: list[int] = [32,64,128], filters_t2: list[int] = [32,64,128], filters_flair: list[int] = [32,64,128], filters_dwi: list[int] = [32,64,128], filters_dwic: list[int] = [32,64,128], filters_shfe: list[int] = [128,128], fusion: FusionType = FusionType.MID_MEAN, train_modality: str = "ALL"):
    r"""
    Build `magnet.MAGNET2` for EzPred
    Args:
        in_ch (int): The number of input channels
        num_classes (int): The number of output classes
        out_main_ch (int): The number of main feature channels
        main_downsample (bool): whether to use the first main downsample layer before branching into `course` or `fine` scales
        filters_t1 (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block for T1 modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        filters_t2 (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block for T2 modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        filters_flair (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block for FLAIR modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        filters_dwi (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block for DWI modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        filters_dwic (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block for DWIC modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        filters_shfe (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `shfe` block for DWIC modality. `len(filters)` indicate the number of `1D conv` blocks for each scale.
        fusion (FusionType): The type of fusion block to fuse the multi-modality features.
    """
    # assert fusion type for MSFE
    assert fusion == FusionType.MID_CONCAT or fusion == FusionType.MID_MEAN, f"Fusion type {fusion} is not supported for MSFE."

    # MSFE for each modalities
    msfe_T1 = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_t1, out_filters=out_filters, main_downsample=main_downsample) # 1x300
    msfe_T2 = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_t2, out_filters=out_filters,main_downsample=main_downsample) # 1x200
    msfe_FLAIR = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_flair, out_filters=out_filters,main_downsample=main_downsample) # 1x200
    msfe_DWI = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_dwi, out_filters=out_filters,main_downsample=main_downsample) # 1x700
    msfe_DWIC = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_dwic, out_filters=out_filters,main_downsample=main_downsample) # 1x499

    # fusion module
    fuse = fusion.load()

    # SHFE for all modalities
    filters_shfe=filters_shfe

    # SHFE for all modalities
    shfe = SHFE(in_ch=filters_shfe[0], out_main_ch=filters_shfe[0]*2, filters=filters_shfe, main_downsample=False)

    # SCH for all modalities
    sch = SCH(mlp_features=filters_shfe[-1] * 2, num_classes=num_classes)

    ## Build target_dict for magnet (Training modalities)
    if train_modality == "ALL": # Training with ALL modalities
        target_dict = {
            0: "T1",
            1: "T2",
            2: "FLAIR",
            3: "DWI",
            4: "DWIC",
        }
    elif train_modality == "FLAIR": # Training with FLAIR Only - Bottom 3 modalities    
        target_dict = {
            2: "FLAIR"
        }
    elif train_modality == "T2": # Training with T2 Only - Bottom 3 modalities    
        target_dict = {
            1: "T2"
        }
    elif train_modality == "T2-FLAIR": # Training with T2 and FLAIR Only - Bottom 3 modalities    
        target_dict = {
            1: "T2",
            2: "FLAIR"
        }
    elif train_modality == "T1-FLAIR-DWIC": # Training with T1, FLAIR and DWIC Only - Top 3 modalities    
        target_dict = {
            0: "T1",
            2: "FLAIR",
            4: "DWIC"
        }
    elif train_modality == "T1-FLAIR-DWI-DWIC": # Training with T1, FLAIR, DWI and DWIC Only - Top 3 modalities    
        target_dict = {
            0: "T1",
            2: "FLAIR",
            3: "DWI",
            4: "DWIC"
        }
    elif train_modality == "T1-T2-FLAIR-DWIC": # Training with T1, T2, FLAIR and DWIC Only - Top 3 modalities    
        target_dict = {
            0: "T1",
            1: "T2",
            2: "FLAIR",
            4: "DWIC"
        }
    elif train_modality == "T1-DWIC": # Training with T1 and DWIC Only - Top modalities    
        target_dict = {
            0: "T1",
            4: "DWIC"
        }
    elif train_modality == "T1-T2-DWIC": # Training with T1, T2 and DWIC Only - Top modalities    
        target_dict = {
            0: "T1",
            1: "T2",
            4: "DWIC"
        }
    elif train_modality == "T1-T2-DWI-DWIC": # Training with T1, T2, DWI and DWIC Only - Top modalities    
        target_dict = {
            0: "T1",
            1: "T2",
            3: "DWI",
            4: "DWIC"
        }
    else:
        raise NotImplementedError("Modality combination training is not needed.")

    model = MAGNET2(msfe_T1, msfe_T2, msfe_FLAIR, msfe_DWI, msfe_DWIC, fusion=fuse, decoder=torch.nn.Sequential(shfe, sch), target_dict=target_dict, return_features=True)

    return model
