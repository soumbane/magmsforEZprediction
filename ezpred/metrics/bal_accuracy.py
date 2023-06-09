from torchmanager.metrics import BinaryConfusionMetric
from torchmanager_core import torch


class BalancedAccuracyScore(BinaryConfusionMetric):

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = tp / (tp + fn + self._eps)
        specificity = tn / (tn + fp + self._eps)

        # Calculate balanced accuracy score
        balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return balanced_acc
