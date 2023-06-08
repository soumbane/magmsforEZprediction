from torchmanager.metrics import BinaryConfusionMetric
from torchmanager_core import torch


class BalancedAccuracyScore(BinaryConfusionMetric):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert predictions to binary (0 or 1) labels
        input = (input > 0.5).float()
        return super().forward(input, target)

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Calculate sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)

        # Calculate balanced accuracy score
        balanced_acc = 0.5 * (sensitivity + specificity).mean()
        return balanced_acc
