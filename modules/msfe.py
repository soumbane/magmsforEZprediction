# Modality-Specific Feature Extractor (MSFE)
import torch
import torch.nn as nn
from torchmanager_core.typing import Any, Enum, Tuple, Sequence


class EZScale(Enum):
    COURSE = "course"
    FINE = "fine"


class conv_block(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, padding: int = 1, bias: bool = False, downsample=None) -> None:
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.lrelu(out1)

        return out1


class msfe(nn.Module):
    r"""
    Args:
        scale (EZScale): The scale of the feature extractor, either `course` or `fine`. `course` indicates 1D convolution with 
        higher receptive field and `fine` indicates 1D convolution with lower receptive field.
    """
    scale: EZScale

    def __init__(self, in_ch: int = 1, out_ch: int = 64, main_downsample: bool = True, scale: EZScale = EZScale.COURSE) -> None:

        self.inplanes = 64

        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.main_downsample = main_downsample
        self.scale = scale

        if self.main_downsample:
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.lrelu = nn.LeakyReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3)

        if self.scale == EZScale.COURSE:
            self.cs_layer_1 = self._cs_make_layer(conv_block, out_ch=64, kernel_size=7, padding=1, bias=False, stride=2)
            self.cs_layer_2 = self._cs_make_layer(conv_block, out_ch=128, kernel_size=7, padding=1, bias=False, stride=2)
            self.cs_layer_3 = self._cs_make_layer(conv_block, out_ch=256, kernel_size=7, padding=1, bias=False, stride=2)

    def _cs_make_layer(self, conv_block, out_ch: int = 64, kernel_size: int = 7, padding: int = 1, bias: bool = False, stride: int = 2):
        downsample = None
        if stride != 1 or self.inplanes != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        layers = []
        layers.append(conv_block(self.inplanes, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, downsample=downsample))

        return nn.Sequential(*layers)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if self.main_downsample:
            x_main = self.conv1(x_in)
            x_main = self.bn1(x_main)
            x_main = self.lrelu(x_main)
            x_main = self.maxpool(x_main)
        else:
            x_main = x_in

        if self.scale == EZScale.COURSE:
            x_cs = self.cs_layer_1(x_main)
            x_cs = self.cs_layer_2(x_cs)
            x_cs = self.cs_layer_3(x_cs)
            x_out = x_cs
        else:
            x_out = x_in

        return x_out


if __name__ == "__main__":

    print("MSFE Module ...")
    msfe_out = msfe(in_ch=1, out_ch=64, main_downsample=True,
                    scale=EZScale.COURSE)

    input_test = torch.randn(1, 1, 300)  # (b, 1, 300)
    out_test = msfe_out(input_test)
    print(out_test.shape)
