# Modality-Specific Feature Extractor (MSFE)
import torch
import torch.nn as nn
from torchmanager_core.typing import Any, Enum, Tuple, Sequence


class EZScale(Enum):
    COURSE = "course"
    FINE = "fine"


'''class fs_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(fs_block, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out'''


class cs_block(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(cs_block, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class msfe(nn.Module):
    r"""
    Args:
        scale (EZScale): The scale of the feature extractor, either `course` or `fine`. `course` indicates 1D convolution with 
        higher receptive field and `fine` indicates 1D convolution with lower receptive field.
    """
    scale: EZScale

    def __init__(self, in_ch: int = 1, out_ch: int = 64, main_downsample: bool = True, scale: EZScale = EZScale.COURSE) -> None:
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
            self.cs_layer_1 = self._cs_make_layer(cs_block, 64, 1, stride=2)
            self.cs_layer_2 = self._cs_make_layer(cs_block, 128, 1, stride=2)
            self.cs_layer_3 = self._cs_make_layer(cs_block, 256, 1, stride=2)

    def _cs_make_layer(self, block, out_ch, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, out_ch * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, out_ch, stride, downsample))
        self.inplanes7 = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes7, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        if self.main_downsample:
            x_main = self.conv1(x_in)
            x_main = self.bn1(x_main)
            x_main = self.lrelu(x_main)
            x_main = self.maxpool(x_main)
        else:
            x_main = x_in

        if self.scale == EZScale.COURSE:




if __name__ == "__main__":

    print("MSFE Module ...")
    msfe_out = msfe(in_ch=1, out_ch=64, main_downsample=True,
                    scale=EZScale.COURSE)

    input_test = torch.randn(1, 1, 300)  # (b, 1, 300)
    out_test = msfe_out(input_test)
    print(out_test.shape)
