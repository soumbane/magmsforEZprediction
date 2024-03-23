from magnet import Manager as _Manager
from torch.optim.optimizer import Optimizer as Optimizer
from torchmanager.callbacks import Frequency
from torchmanager.data import DataLoader, Dataset
from torchmanager.losses.loss import Loss
from torchmanager.metrics.metric import Metric
from torchmanager_core import errors, deprecated
from torchmanager_core.typing import Module, Any, Optional, Union
import torch


# @deprecated('v0.2', 'v1.0')
# class Manager(_Manager[Module]):
#     pass

class Manager(_Manager[Module]):
    combined_tensors: list[torch.Tensor]

    def __init__(self, model: Module, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[Loss, dict[str, Loss]]] = None, metrics: dict[str, Metric] = {}, target_freq: Optional[Frequency] = None) -> None:
        super().__init__(model, optimizer, loss_fn, metrics, target_freq)
        self.combined_tensors = []

    @torch.no_grad()
    def test(self, dataset: Union[DataLoader[Any], Dataset[Any], dict[Optional[int], Union[DataLoader[Any], Dataset[Any]]]], show_verbose: bool = False, return_combined_tensors: bool = False, **kwargs: Any) -> Union[tuple[dict[str, float], torch.Tensor], dict[str, float]]:
        self.combined_tensors.clear()
        summary = super().test(dataset, show_verbose, **kwargs)
        mean_combined_tensors = torch.cat(self.combined_tensors, dim=0).mean(0)
        return (summary, mean_combined_tensors) if return_combined_tensors else summary

    def test_step(self, x_test: Any, y_test: Any) -> dict[str, float]:
        """
        A single testing step

        - Parameters:
            - x_train: The testing data in `torch.Tensor`
            - y_train: The testing label in `torch.Tensor`
        - Returns: A `dict` of validation summary
        """
        # forward pass
        y, _ = self.forward(x_test, y_test)

        # extract y
        if not self.model.training:
            y, y_combined = y
            if isinstance(y_combined, list):
                y_combined = y_combined[0]
            assert isinstance(y_combined, torch.Tensor), "The combined tensor should be a `torch.Tensor`."
            self.combined_tensors.append(y_combined)

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"):
                name = name.replace("val_", "")
            elif "loss" in name:
                continue
            try:
                fn(y, y_test)
            except Exception as metric_error:
                runtime_error = errors.MetricError(name)
                raise runtime_error from metric_error
            
        return self.summary
