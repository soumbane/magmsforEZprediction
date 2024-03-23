from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any, Union
from magnet.nn.shared import FeaturedData


class FeaturedMetric(Metric):
    def __call__(self, input: Union[FeaturedData, torch.Tensor], target: Any) -> torch.Tensor:
        if isinstance(input, FeaturedData):
            input = input.out[:, 0, ...]
        return super().__call__(input, target)
