from torchmanager.metrics import BinaryConfusionMetric
from torchmanager_core import torch
from torchmanager_core.typing import Optional


class BalancedAccuracyScore(BinaryConfusionMetric):
    conf_met: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        tp = self.conf_met[0, 0]
        fn = self.conf_met[0, 1]
        fp = self.conf_met[1, 0]
        tn = self.conf_met[1, 1]
        return self.forward_metric(tp, tn, fp, fn)

    @property
    def results(self) -> Optional[torch.Tensor]:
        return self.conf_met

    def __init__(self, dim: int = -1, *, eps: float = 1e-7, target: Optional[str] = None):
        super().__init__(dim, eps=eps, target=target)
        self.conf_met = torch.nn.Parameter(torch.zeros((2,2)))

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        self.conf_met[0, 0] += tp
        self.conf_met[0, 1] += fn
        self.conf_met[1, 0] += fp
        self.conf_met[1, 1] += tn

        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = tp / (tp + fn + self._eps)
        specificity = tn / (tn + fp + self._eps)

        # Calculate balanced accuracy score
        balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return balanced_acc

    def reset(self) -> None:
        self.conf_met = torch.nn.Parameter(torch.zeros((2,2)))
        return super().reset()
