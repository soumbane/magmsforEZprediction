from torchmanager.metrics import BinaryConfusionMetric, SparseCategoricalAccuracy
from torchmanager_core import torch
from torchmanager_core.typing import Optional

from .featured import FeaturedMetric


class AccuracyScore(SparseCategoricalAccuracy, FeaturedMetric):
    def __init__(self, dim: int = -1, *, target: Optional[str] = None) -> None:
        SparseCategoricalAccuracy.__init__(self, dim, target=target)


class BalancedAccuracyScore(BinaryConfusionMetric, FeaturedMetric):
    conf_met: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        tp = self.conf_met[0, 0]
        fn = self.conf_met[0, 1]
        fp = self.conf_met[1, 0]
        tn = self.conf_met[1, 1]
        return self.calculate_bal_acc(tp, tn, fp, fn)

    @property
    def results(self) -> Optional[torch.Tensor]:
        return self.conf_met

    def __init__(self, dim: int = -1, *, eps: float = 1e-7, target: Optional[str] = None):
        BinaryConfusionMetric.__init__(self, dim, class_index=0, eps=eps, target=target)
        self.conf_met = torch.nn.Parameter(torch.zeros((2,2)), requires_grad=False)

    def calculate_bal_acc(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = tp / (tp + fn + self._eps)
        specificity = tn / (tn + fp + self._eps) if torch.sum(tn + fp) > 0 else sensitivity

        # Calculate balanced accuracy score
        balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return balanced_acc

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        self.conf_met[0, 0] += tp
        self.conf_met[0, 1] += fn
        self.conf_met[1, 0] += fp
        self.conf_met[1, 1] += tn
        return self.calculate_bal_acc(tp, tn, fp, fn)

    def reset(self) -> None:
        self.conf_met *= 0
        return super().reset()
    

class SensitivityScore(BinaryConfusionMetric, FeaturedMetric):
    conf_met: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        tp = self.conf_met[0, 0]
        fn = self.conf_met[0, 1]
        fp = self.conf_met[1, 0]
        tn = self.conf_met[1, 1]
        return self.calculate_sensitivity(tp, tn, fp, fn)

    @property
    def results(self) -> Optional[torch.Tensor]:
        return self.conf_met

    def __init__(self, dim: int = -1, *, eps: float = 1e-7, target: Optional[str] = None):
        BinaryConfusionMetric.__init__(self, dim, class_index=0, eps=eps, target=target)
        self.conf_met = torch.nn.Parameter(torch.zeros((2,2)), requires_grad=False)

    def calculate_sensitivity(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = tp / (tp + fn + self._eps)
        # specificity = tn / (tn + fp + self._eps)

        # # Calculate balanced accuracy score
        # balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return sensitivity.mean()

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        self.conf_met[0, 0] += tp
        self.conf_met[0, 1] += fn
        self.conf_met[1, 0] += fp
        self.conf_met[1, 1] += tn
        return self.calculate_sensitivity(tp, tn, fp, fn)

    def reset(self) -> None:
        self.conf_met *= 0
        return super().reset()
    

class SpecificityScore(BinaryConfusionMetric, FeaturedMetric):
    conf_met: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        tp = self.conf_met[0, 0]
        fn = self.conf_met[0, 1]
        fp = self.conf_met[1, 0]
        tn = self.conf_met[1, 1]
        return self.calculate_specificity(tp, tn, fp, fn)

    @property
    def results(self) -> Optional[torch.Tensor]:
        return self.conf_met

    def __init__(self, dim: int = -1, *, eps: float = 1e-7, target: Optional[str] = None):
        BinaryConfusionMetric.__init__(self, dim, class_index=0, eps=eps, target=target)
        self.conf_met = torch.nn.Parameter(torch.zeros((2,2)), requires_grad=False)

    def calculate_specificity(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        # sensitivity = tp / (tp + fn + self._eps)
        specificity = tn / (tn + fp + self._eps)

        # # Calculate balanced accuracy score
        # balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return specificity.mean()

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        self.conf_met[0, 0] += tp
        self.conf_met[0, 1] += fn
        self.conf_met[1, 0] += fp
        self.conf_met[1, 1] += tn
        return self.calculate_specificity(tp, tn, fp, fn)

    def reset(self) -> None:
        self.conf_met *= 0
        return super().reset()
