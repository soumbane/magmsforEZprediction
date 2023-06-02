# Modality-Specific Feature Extractor (MSFE)
import torch
import torch.nn as nn
from torchmanager_core.typing import Any, Enum, Sequence, Optional


class EZScale(Enum):
    COURSE = "course"
    FINE = "fine"


class conv_block(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, padding: int = 1, bias: bool = False, downsample=None, scale: EZScale = EZScale.COURSE) -> None:
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if self.scale == EZScale.COURSE:
            d = residual.shape[2] - out.shape[2]
            out = residual[:, :, 0:-d] + out
        elif self.scale == EZScale.FINE:
            out += residual
        else:
            raise NotImplementedError
        
        out = self.lrelu(out)

        return out


class msfe(nn.Module):
    r"""
    Args:
        scale (EZScale): The scale of the feature extractor, either `course` or `fine`. `course` indicates 1D convolution with 
        higher receptive field and `fine` indicates 1D convolution with lower receptive field.
    """
    scale: EZScale

    def __init__(self, in_ch: int = 1, out_main_ch: int = 32, filters: list[int] = [32,64,128], main_downsample: bool = True, scale: EZScale = EZScale.COURSE) -> None:

        self.inplanes = 32

        super().__init__()
        self.in_ch = in_ch
        self.out_main_ch = out_main_ch
        self.main_downsample = main_downsample
        self.scale = scale

        if self.main_downsample:
            self.conv1 = nn.Conv1d(in_ch, out_main_ch, kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_main_ch)
            self.lrelu = nn.LeakyReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3)

        if self.scale == EZScale.COURSE:
            self.cs_layers = []
            for i in range(len(filters)):
                self.cs_layers.append(self._make_conv_layer(conv_block, out_ch=filters[i], kernel_size=7, padding=1, bias=False, stride=1, scale=self.scale))
            self.cs_layers_f = nn.Sequential(*self.cs_layers)
    
        elif self.scale == EZScale.FINE:
            self.fs_layers = []
            for i in range(len(filters)):
                self.fs_layers.append(self._make_conv_layer(conv_block, out_ch=filters[i], kernel_size=3, padding=1, bias=False, stride=1, scale=self.scale))
            self.fs_layers_f = nn.Sequential(*self.fs_layers)

        else:
            raise NotImplementedError


    def _make_conv_layer(self, conv_block, out_ch: int = 32, kernel_size: int = 7, padding: int = 1, bias: bool = False, stride: int = 2, scale: EZScale = EZScale.COURSE):
        downsample = None
        if stride != 1 or self.inplanes != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        layers = []
        layers.append(conv_block(self.inplanes, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, downsample=downsample, scale=scale))
        self.inplanes = out_ch

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
            x_out = self.cs_layers_f(x_main)

        elif self.scale == EZScale.FINE:
            x_out = self.fs_layers_f(x_main)

        else:
            raise NotImplementedError

        return x_out


if __name__ == "__main__":

    print("MSFE Module ...")
    msfe_out = msfe(in_ch=1, out_main_ch=32, filters=[32,64,128], main_downsample=True, scale=EZScale.COURSE)

    input_test = torch.randn(1, 1, 200)  # (b, 1, 200)
    out_test = msfe_out(input_test)
    print(out_test.shape)
