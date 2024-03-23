# Shared Classification Head (SCH)
import torch
import torch.nn as nn
from typing import Union


class SCH(nn.Module):
    r"""
    Shared Classification Head
    Args:
        mlp_features: input features for the fully connected layer - this is the concatenation of all the feature maps obtained from the `shfe` course and fine scale modules
        num_classes: the final output number of classes of the MAG-MS model
    """

    def __init__(self, mlp_features: int = 256, num_classes: int = 2) -> None:
        super().__init__()

        self.adaptive_average_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Linear(mlp_features, num_classes)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, tuple[torch.Tensor,torch.Tensor]]:
        r"""
        Args:
            x_cs: course scale input
            x_fs: fine scale input
        """
        x_cs, x_fs = x

        x_cs_avg = self.adaptive_average_pool(x_cs)

        x_fs_avg = self.adaptive_average_pool(x_fs)

        x_combined = torch.cat((x_cs_avg,x_fs_avg), dim=1).squeeze(dim=2)

        x_final = self.mlp(x_combined)

        # return self.softmax(x_final)
        # return x_final
        if self.training:
            return x_final
        else:
            return x_final, x_combined


if __name__ == "__main__":

    print("SCH Module ...")
    sch_out = SCH(mlp_features=256, num_classes=2)

    input_test_fs = torch.randn(1, 128, 200)  # (b, 128, 200)
    input_test_cs = torch.randn(1, 128, 176)  # (b, 128, 176)
    out_test = sch_out(input_test_cs, input_test_fs)
    print(out_test.shape)
