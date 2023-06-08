# Shared Feature Extractor (SHFE)
import torch
from .msfe import MSFEScale, EZScale

SHFEScale = MSFEScale # shfe is alias of msfe


class SHFE(torch.nn.Module):
    cs: SHFEScale
    fs: SHFEScale

    def __init__(self, in_ch: int = 1, out_main_ch: int = 32, filters: list[int] = [32,64,128], main_downsample: bool = False) -> None:
        super().__init__()
        self.cs = MSFEScale(in_ch, out_main_ch, filters=filters, main_downsample=main_downsample, scale=EZScale.COURSE)
        self.cs_conv = torch.nn.Conv1d(filters[-1], filters[-1], kernel_size=1)
        self.fs = MSFEScale(in_ch, out_main_ch, filters=filters, main_downsample=main_downsample, scale=EZScale.FINE)
        self.fs_conv = torch.nn.Conv1d(filters[-1], filters[-1], kernel_size=1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # unpack data
        x_cs, x_fs = x

        # course block
        y_cs = self.cs(x_cs)

        # fine block
        y_fs = self.fs(x_fs)
        return y_cs, y_fs



if __name__ == "__main__":

    print("SHFE Module ...")
    shfe_out = SHFEScale(in_ch=1, out_main_ch=32, filters=[32,64,128], main_downsample=False, scale=EZScale.FINE)

    input_test = torch.randn(1, 1, 200)  # (b, 1, 200)
    out_test = shfe_out(input_test)
    print(out_test.shape)
    