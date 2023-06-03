# Shared Feature Extractor (SHFE)
import torch
from msfe import msfe, EZScale

shfe = msfe # shfe is alias of msfe


if __name__ == "__main__":

    print("SHFE Module ...")
    shfe_out = shfe(in_ch=1, out_main_ch=32, filters=[32,64,128], main_downsample=False, scale=EZScale.FINE)

    input_test = torch.randn(1, 1, 200)  # (b, 1, 200)
    out_test = shfe_out(input_test)
    print(out_test.shape)
    