from torchmanager.metrics import ConfusionMetrics as _ConfMet
from torchmanager_core import torch
from torchmanager_core.typing import Any, Optional, Union
from magnet.nn.shared import FeaturedData


class ConfusionMetrics(_ConfMet):
    @property
    def result(self) -> torch.Tensor:
        return torch.tensor(torch.nan)

    @property
    def results(self) -> Optional[torch.Tensor]:
        r = super().results
        return r.sum(0) if r is not None else r
    
    def __call__(self, input: Union[FeaturedData, torch.Tensor], target: Any) -> torch.Tensor:
        if isinstance(input, FeaturedData):
            input = input.out[:, 0, ...]
        assert isinstance(input, torch.Tensor), "Input must be a tensor."
        input = input.softmax(1) # (b, c) -> (b)
        return super().__call__(input, target)

    def forward(self, input: torch.Tensor, target: Any) -> torch.Tensor:
        return input.softmax(1)
