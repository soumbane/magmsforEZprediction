import torch
from magnet import nn, MAGNET2

from ezpred.nn import MSFE, SHFE, SCH


def load_magms_ezpred(in_ch: int = 1, out_main_ch: int = 32, main_downsample: bool = False, filters_T1: list[int] = [32,64,128], filters_T2: list[int] = [32,64,128], filters_FLAIR: list[int] = [32,64,128], filters_DWI: list[int] = [32,64,128,256], filters_DWIC: list[int] = [32,64,128], filters_SHFE: list[int] = [128,256,512], mlp_features: int = 256, num_classes: int = 2) -> MAGNET2:
    # MSFE for T1
    msfe_T1 = MSFE(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T1, main_downsample=main_downsample)

    # MSFE for T2
    msfe_T2 = MSFE(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T2, main_downsample=main_downsample)

    # MSFE for FLAIR
    msfe_FLAIR = MSFE(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_FLAIR, main_downsample=main_downsample)

    # MSFE for DWI
    msfe_DWI = MSFE(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWI, main_downsample=main_downsample)

    # MSFE for DWIC
    msfe_DWIC = MSFE(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWIC, main_downsample=main_downsample)

    # fusion module
    fuse = nn.fusion.MidFusion()

    # SHFE for all modalities
    shfe = SHFE(in_ch=128, out_main_ch=128, filters=filters_SHFE, main_downsample=False)

    # SCH for all modalities
    sch = SCH(mlp_features=mlp_features, num_classes=num_classes)

    # build magnet
    magnet = MAGNET2(msfe_T1, msfe_T2, msfe_FLAIR, msfe_DWI, msfe_DWIC, fusion=fuse, decoder=torch.nn.Sequential(shfe, sch))
    return magnet