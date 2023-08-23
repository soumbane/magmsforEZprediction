import argparse, magnet, os, torch, torchmanager
from torchmanager_core import view
from typing import Union

from ezpred import DESCRIPTION
from .basic import Configs as _Configs


class Configs(_Configs):
    """
    The testing configurations
    Args:
        batch_size (int): The number of batch size.
        data_dir (str): The root directory of the dataset.
        device (torch.device): The target device to run testing.
        show_verbose (bool): Flag to show progress bar during testing.
        use_multi_gpus (bool): Flag to use multiple GPUs.
    """
    batch_size: int
    data_dir: str
    device: torch.device
    model: str
    node_num: int
    fold_no: str
    show_verbose: bool
    use_multi_gpus: bool

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device)
        self.model = os.path.normpath(self.model)

        # check format
        assert self.batch_size > 0, f"Batch size must be a positive number, got {self.batch_size}."
        # assert self.fold_no in range(1, 6), f"Fold number must be in range of [1,5], got {self.fold_no}."

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # main arguments
        parser.add_argument("data_dir", type=str, help="The root directory of the dataset.")
        parser.add_argument("model", type=str, help="The trained model path.")
        parser.add_argument("-node", "--node_num", type=int, required=False, help="The node number for evaluation, must be specified.")

        # testing arguments
        testing_args = parser.add_argument_group("Testing arguments")
        testing_args.add_argument("-b", "--batch_size", type=int, default=1, help="The number of batch size, default is 1.")
        testing_args.add_argument("--fold_no", type=str, required=False, default=1, help="Fold number for validation, default is 1.")
        testing_args.add_argument("--show_verbose", action="store_true", default=False, help="The flag to show probress bar during testing.")
        _Configs.get_arguments(testing_args)

        # device arguments
        device_args = parser.add_argument_group("Device arguments")
        device_args.add_argument("--device", type=str, default="cuda", help="The target device to run testing, default is `cuda`.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="The flag to use multi GPUs during testing.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}, magnet={magnet.VERSION}")

    def show_settings(self) -> None:
        view.logger.info(f"Dataset: data_dir={self.data_dir}, node={self.node_num}")
        view.logger.info(f"Fold for testing: fold_no={self.fold_no}")
        view.logger.info(f"Output: model={self.model}")
        view.logger.info(f"testing: batch_size={self.batch_size}, show_verbose={self.show_verbose}")
        view.logger.info(f"Device: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
