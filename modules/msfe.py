# Modality-Specific Feature Extractor (MSFE)
import torch
import torch.nn as nn
from torchmanager_core.typing import Enum


class EZScale(Enum):
    COURSE = "course"
    FINE = "fine"


class conv_block(nn.Module):
    r"""
    Convolution block consisting of `1Dconv->BN->LReLU->1Dconv->BN->LReLU` with input residual connection added after the second batchnorm 
    Args:
        in_ch (int): The number of input channels 
        out_ch (int): The number of output channels 
        kernel_size (int): The kernel size of each 1D `conv` layer in either `course (kernel_size=7)` scale or `fine (kernel_size=3)` scale 
        stride (int): The stride of the 1D conv kernel
        padding (int): The padding of the 1D conv kernel
        bias (bool): whether to add bias term
        downsample (nn.Sequential): The downsample layers to add to the input residual connection to the output of conv blocks
        scale (EZScale): The scale of the feature extractor, either `course` or `fine`. `course` indicates 1D convolution with higher receptive field and `fine` indicates 1D convolution with lower receptive field.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, padding: int = 1, bias: bool = False, downsample = None, scale: EZScale = EZScale.COURSE) -> None:
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: input of each module consisting of `1Dconv->BN->LReLU->1Dconv->BN->LReLU`
        """
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
    Modality-Specific Feature Extractor with both `course` and `fine` scales. `course` scale indicates 1D conv with kernel_size of 7 and `fine` scale indicates 1D conv with kernel_size of 3 
    Args:
        in_ch (int): The number of input channels to each `msfe` block
        out_main_ch (int): The number of output channels of the first main downsample layer of each `msfe` block. The first main downsample layer is applied before branching into `course` and `fine` scales
        filters (list[int]): The output channels of each 1D `conv` layer blocks in either `course` scale or `fine` scale of each `msfe` block. `len(filters)` indicate the number of `1D conv` blocks for each scale. 
        main_downsample (bool): whether to use the first main downsample layer before branching into `course` or `fine` scales
        scale (EZScale): The scale of the feature extractor, either `course` or `fine`. `course` indicates 1D convolution with higher receptive field and `fine` indicates 1D convolution with lower receptive field.
    """
    scale: EZScale

    def __init__(self, in_ch: int = 1, out_main_ch: int = 32, filters: list[int] = [32,64,128], main_downsample: bool = False, scale: EZScale = EZScale.COURSE) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_main_ch = out_main_ch
        self.main_downsample = main_downsample
        self.scale = scale
        self.inplanes = self.out_main_ch

        if self.main_downsample:
            self.conv1 = nn.Conv1d(in_ch, out_main_ch, kernel_size=3, stride=2, padding=2, bias=False)
            self.bn1 = nn.BatchNorm1d(out_main_ch)
            self.lrelu = nn.LeakyReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_main_ch, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_main_ch)

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


    def _make_conv_layer(self, conv_block, out_ch: int = 32, kernel_size: int = 7, padding: int = 1, bias: bool = False, stride: int = 2, scale: EZScale = EZScale.COURSE) -> nn.Sequential:
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
        r"""
        Args:
            x_in (1xN): input of each `msfe` module
        """
        if self.main_downsample:
            x_main = self.conv1(x_in)
            x_main = self.bn1(x_main)
            x_main = self.lrelu(x_main)
            x_main = self.maxpool(x_main)
        else:
            x_main = self.conv1(x_in)
            x_main = self.bn1(x_main)

        if self.scale == EZScale.COURSE:
            x_out = self.cs_layers_f(x_main)

        elif self.scale == EZScale.FINE:
            x_out = self.fs_layers_f(x_main)

        else:
            raise NotImplementedError

        return x_out


if __name__ == "__main__":

    print("MSFE Module ...")
    msfe_out = msfe(in_ch=1, out_main_ch=32, filters=[32,64,128], main_downsample=False, scale=EZScale.FINE)

    input_test = torch.randn(1, 1, 200)  # (b, 1, 200)
    out_test = msfe_out(input_test)
    print(out_test.shape)
