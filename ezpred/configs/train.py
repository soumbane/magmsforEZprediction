import argparse, magnet, os, torch, torchmanager
from torchmanager_core import view
from typing import Optional, Union

from ezpred import DESCRIPTION
from .basic import Configs as _Configs


class Configs(_Configs):
    """
    The training configurations
    Args:
        batch_size (int): The number of batch size.
        learning_rate (float): The learning rate for training
        data_dir (str): The root directory of the dataset.
        device (torch.device): The target device to run training.
        epochs (int): The number of training epochs.
        show_verbose (bool): Flag to show progress bar during training.
        use_multi_gpus (bool): Flag to use multiple GPUs.
    """
    batch_size: int
    data_dir: str
    device: torch.device
    epochs: int
    learning_rate: float
    node_num: int
    output_model: str
    seed: Optional[int]
    show_verbose: bool
    use_multi_gpus: bool

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device)
        self.output_model = os.path.normpath(self.output_model)
        output_model_dir = os.path.dirname(self.output_model)
        os.makedirs(output_model_dir, exist_ok=True)

        # check format
        assert self.batch_size > 0, f"Batch size must be a positive number, got {self.batch_size}."
        assert self.learning_rate > 0, f"Learning rate must be positive, got {self.learning_rate}."
        assert self.epochs > 0, f"Number of epochs must be a positive number, got {self.epochs}."
        assert self.node_num in range(1, 999), f"Node number must be in range of [1, 998], got {self.node_num}."
        if self.seed is not None:
            assert torchmanager.version > "v1.1", f"Torchmanager version 1.2 required to freeze seed, {torchmanager.version} installed."
            assert self.seed >= 0, f"Seed must be a non-negative number, got {self.seed}."

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # main arguments
        parser.add_argument("data_dir", type=str, help="The root directory of the dataset.")
        parser.add_argument("output_model", type=str, help="The output directory for the final trained (last) model.")
        parser.add_argument("-node", "--node_num", type=int, required=True, help="The node number to train, must be specified.")

        # training arguments
        training_args = parser.add_argument_group("Training arguments")
        training_args.add_argument("-e", "--epochs", type=int, default=100, help="The number of training epochs, default is 100.")
        training_args.add_argument("-b", "--batch_size", type=int, default=1, help="The number of batch size, default is 1.")
        training_args.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="Learning rate, default is 5e-5.")
        training_args.add_argument("--seed", type=int, default=None, help="The random seed for training (torchmanager 1.2 required if given), default is `None`.")
        training_args.add_argument("--show_verbose", action="store_true", default=False, help="The flag to show probress bar during training.")
        _Configs.get_arguments(training_args)

        # device arguments
        device_args = parser.add_argument_group("Device arguments")
        device_args.add_argument("--device", type=str, default="cuda", help="The target device to run training, default is `cuda`.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="The flag to use multi GPUs during training.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}, magnet={magnet.VERSION}")

    def show_settings(self) -> None:
        view.logger.info(f"Dataset: data_dir={self.data_dir}, node={self.node_num}")
        view.logger.info(f"Output: output_model={self.output_model}")
        view.logger.info(f"Training: epochs={self.epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, seed={self.seed}, show_verbose={self.show_verbose}")
        view.logger.info(f"Device: device={self.device}, use_multi_gpus={self.use_multi_gpus}")