from enum import Enum
from magnet.nn.fusion import Fusion, MidFusion, MidSingleFusion

from .fuse import MidConcat, MidSingleConcat


class FusionType(Enum):
    MID_CONCAT = "mid_concat"
    MID_MEAN = "mid_mean"
    MID_SINGLE_CONCAT = "mid_single_concat"
    MID_SINGLE_MEAN = "mid_single_mean"

    def load(self) -> Fusion:
        if self == FusionType.MID_CONCAT:
            return MidConcat()
        elif self == FusionType.MID_MEAN:
            return MidFusion()
        elif self == FusionType.MID_SINGLE_CONCAT:
            return MidSingleConcat()
        elif self == FusionType.MID_SINGLE_MEAN:
            return MidSingleFusion()
        else:
            raise TypeError(f"Fusion type {self} currently not supported.")
