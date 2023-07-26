import torch
from magnet import MAGNET2
from .networks import Basic

from .nn import MSFE, SHFE, SCH
from .nn.fusion import FusionType

# def build(num_classes: int = 2, /, out_main_ch: int = 32, main_downsample: bool = False, *, filters_t1: list[int] = [32,64,128], filters_t2: list[int] = [32,64,128], filters_flair: list[int] = [32,64,128], filters_dwi: list[int] = [32,64,128], filters_dwic: list[int] = [32,64,128], filters_shfe: list[int] = [128,256,512], fusion: FusionType = FusionType.MID_CONCAT) -> Basic: # for Min-Hee's code

def build(num_classes: int = 2, /, out_main_ch: int = 32, main_downsample: bool = False, *, filters_t1: list[int] = [32,64,128], filters_t2: list[int] = [32,64,128], filters_flair: list[int] = [32,64,128], filters_dwi: list[int] = [32,64,128], filters_dwic: list[int] = [32,64,128], filters_shfe: list[int] = [128,128], fusion: FusionType = FusionType.MID_MEAN) -> MAGNET2[MSFE]:
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

    filters_msfe=[64,128,256] # deep model
    # filters_msfe=[8,16] # shallow model

    # MSFE for each modalities
    msfe_T1 = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_msfe, main_downsample=main_downsample, padding=400)
    msfe_T2 = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_msfe, main_downsample=main_downsample, padding=500)
    msfe_FLAIR = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_msfe, main_downsample=main_downsample, padding=500)
    msfe_DWI = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_msfe, main_downsample=main_downsample)
    msfe_DWIC = MSFE(in_ch=1, out_main_ch=out_main_ch, filters=filters_msfe, main_downsample=main_downsample, padding=201)

    # fusion module
    fuse = fusion.load()

    filters_shfe=[filters_msfe[-1],512] # deep model
    # filters_shfe=[filters_msfe[-1],32] # shallow model

    # SHFE for all modalities
    shfe = SHFE(in_ch=filters_msfe[-1], out_main_ch=filters_msfe[-1], filters=filters_shfe, main_downsample=False)

    # SCH for all modalities
    sch = SCH(mlp_features=filters_shfe[-1] * 2, num_classes=num_classes)
    # sch = SCH(mlp_features=filters_msfe[-1] * 10, num_classes=num_classes) # for Min-Hee's code

    # build magnet
    target_dict = {
        0: "T1",
        1: "T2",
        2: "FLAIR",
        3: "DWI",
        4: "DWIC",
    }

    model = MAGNET2(msfe_T1, msfe_T2, msfe_FLAIR, msfe_DWI, msfe_DWIC, fusion=fuse, decoder=torch.nn.Sequential(shfe, sch), target_dict=target_dict, return_features=True)

    # model = Basic(msfe_T1, msfe_T2, msfe_FLAIR, msfe_DWI, msfe_DWIC, fusion=fuse, sch=sch) # for Min-Hee's code

    # build losses
    return model