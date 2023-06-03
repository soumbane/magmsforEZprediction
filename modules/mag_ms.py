# The Modality Agnostic (MAG) framework combining MSFE, SHFE and SCH modules

import torch
import torch.nn as nn
from torchmanager_core.typing import Any

from msfe import msfe, EZScale
from shfe import shfe
from sch import sch

class mag_ms(nn.Module):
    def __init__(self, in_ch: int = 1, out_main_ch: int = 32, main_downsample: bool = False, filters_T1: list[int] = [32,64,128], filters_T2: list[int] = [32,64,128], filters_FLAIR: list[int] = [32,64,128], filters_DWI: list[int] = [32,64,128,256], filters_DWIC: list[int] = [32,64,128], filters_SHFE: list[int] = [128,256,512], mlp_features: int = 256, num_classes: int = 2) -> None:
        super().__init__()

        # MSFE for T1
        self.msfe_cs_T1 = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T1, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.msfe_fs_T1 = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T1, main_downsample=main_downsample, scale=EZScale.FINE)

        # MSFE for T2
        self.msfe_cs_T2 = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T2, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.msfe_fs_T2 = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_T2, main_downsample=main_downsample, scale=EZScale.FINE)

        # MSFE for FLAIR
        self.msfe_cs_FLAIR = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_FLAIR, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.msfe_fs_FLAIR = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_FLAIR, main_downsample=main_downsample, scale=EZScale.FINE)

        # MSFE for DWI
        self.msfe_cs_DWI = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWI, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.msfe_fs_DWI = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWI, main_downsample=main_downsample, scale=EZScale.FINE)

        # MSFE for DWIC
        self.msfe_cs_DWIC = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWIC, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.msfe_fs_DWIC = msfe(in_ch=in_ch, out_main_ch=out_main_ch, filters=filters_DWIC, main_downsample=main_downsample, scale=EZScale.FINE)

        # SHFE for all modalities
        self.shfe_cs = shfe(in_ch=128, out_main_ch=128, filters=filters_SHFE, main_downsample=False, scale=EZScale.COURSE)
        self.shfe_fs = shfe(in_ch=128, out_main_ch=128, filters=filters_SHFE, main_downsample=False, scale=EZScale.FINE)

        # SCH for all modalities
        self.sch = sch(mlp_features=mlp_features, num_classes=num_classes)

    def forward(self, x_in: list[torch.Tensor]) -> Any:

        # Extract the different modalities
        x_T1 = x_in[0]
        x_T2 = x_in[1]
        x_FLAIR = x_in[2]
        x_DWI = x_in[3]
        x_DWIC = x_in[4]

        # Apply the MSFE blocks (both course and fine scales) to the modalities
        # MSFE for T1
        x_T1_cs = self.msfe_cs_T1(x_T1)
        x_T1_fs = self.msfe_fs_T1(x_T1)

        # MSFE for T2
        x_T2_cs = self.msfe_cs_T2(x_T2)
        x_T2_fs = self.msfe_fs_T2(x_T2)

        # MSFE for FLAIR
        x_FLAIR_cs = self.msfe_cs_FLAIR(x_FLAIR)
        x_FLAIR_fs = self.msfe_fs_FLAIR(x_FLAIR)

        # MSFE for DWI
        x_DWI_cs = self.msfe_cs_DWI(x_DWI)
        x_DWI_fs = self.msfe_fs_DWI(x_DWI)

        # MSFE for DWIC
        x_DWIC_cs = self.msfe_cs_DWIC(x_DWIC)
        x_DWIC_fs = self.msfe_fs_DWIC(x_DWIC)

        # SHFE (course scale) for all modalities
        x_shared_T1_cs = self.shfe_cs(x_T1_cs)
        x_shared_T2_cs = self.shfe_cs(x_T2_cs)
        x_shared_FLAIR_cs = self.shfe_cs(x_FLAIR_cs)
        x_shared_DWI_cs = self.shfe_cs(x_DWI_cs)
        x_shared_DWIC_cs = self.shfe_cs(x_DWIC_cs)

        # SHFE (fine scale) for all modalities
        x_shared_T1_fs = self.shfe_fs(x_T1_fs)
        x_shared_T2_fs = self.shfe_fs(x_T2_fs)
        x_shared_FLAIR_fs = self.shfe_fs(x_FLAIR_fs)
        x_shared_DWI_fs = self.shfe_fs(x_DWI_fs)
        x_shared_DWIC_fs = self.shfe_fs(x_DWIC_fs)

        # SHFE (course scale) for fused modalities: TO DO
        x_shared_fused_cs = None

        # SHFE (fine scale) for fused modalities: TO DO
        x_shared_fused_fs = None

        # SCH for fused modalities
        x_out_all = self.sch(x_shared_fused_cs, x_shared_fused_fs)

        # SCH for individual modalities
        x_out_T1 = self.sch(x_shared_T1_cs, x_shared_T1_fs)
        x_out_T2 = self.sch(x_shared_T2_cs, x_shared_T2_fs)
        x_out_FLAIR = self.sch(x_shared_FLAIR_cs, x_shared_FLAIR_fs)
        x_out_DWI = self.sch(x_shared_DWI_cs, x_shared_DWI_fs)
        x_out_DWIC = self.sch(x_shared_DWIC_cs, x_shared_DWIC_fs)

        return x_out_all, x_out_T1, x_out_T2, x_out_FLAIR, x_out_DWI, x_out_DWIC


if __name__ == "__main__":

    print("MAG-MS Module ...")
    mag_ms_out = mag_ms(in_ch=1, out_main_ch=32, main_downsample=False, filters_T1=[32,64,128], filters_T2=[32,64,128], filters_FLAIR=[32,64,128], filters_DWI=[32,64,128,256], filters_DWIC=[32,64,128], filters_SHFE=[128,256,512], mlp_features=256, num_classes=2)

    input_test_T1 = torch.randn(1, 1, 300)  # (b, 1, 300)
    input_test_T2 = torch.randn(1, 1, 200)  # (b, 1, 200)
    input_test_FLAIR = torch.randn(1, 1, 200)  # (b, 1, 200)
    input_test_DWI = torch.randn(1, 1, 700)  # (b, 1, 700)
    input_test_DWIC = torch.randn(1, 1, 499)  # (b, 1, 499)

    out_all, out_T1, out_T2, out_FLAIR, out_DWI, out_DWIC = mag_ms_out([input_test_T1, input_test_T2, input_test_FLAIR, input_test_DWI, input_test_DWIC])

    print(out_all.shape)
    print(out_T1.shape)
    print(out_T2.shape)
    print(out_FLAIR.shape)
    print(out_DWI.shape)
    print(out_DWIC.shape)
