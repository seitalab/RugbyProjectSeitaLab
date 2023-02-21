from argparse import Namespace
from typing import Iterable, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import Trial
from torchinfo import summary
import torch.backends.cudnn as cudnn

from codes.supports.storer import Storer
from codes.data.dataloader import prepare_dataloader
from codes.architectures.model import prepare_model

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

class BaseTrainer:

    def __init__(
        self,
        args: Namespace,
        save_dir: str,
        info_string: Optional[str] = None
    ) -> None:
        """
        Args:
            args (Namespace):
            save_dir (str): Directory to output model weights and parameter info.
            info_string (Optional[str]): String info to print during training.
        Returns:
            None
        """

        self.args = args

        self.storer = Storer(save_dir)
        self.model = None

        self.info_string = info_string
        
        self.best = None # Overwritten during training.

    def prepare_dataloader(self) -> Tuple[Iterable, Iterable]:
        """
        Args:
            None
        Returns:
            train_loader (Iterable):
            valid_loader (Iterable):
        """

        # Prepare dataloader.
        train_loader, valid_loader, _ = prepare_dataloader(self.args)
        return train_loader, valid_loader

    def set_lossfunc(self, weights: Optional[np.ndarray]=None) -> None:
        """
        Args:
            weights (Optional[np.ndarray]): 
        Returns:
            None
        """
        assert self.model is not None
        if weights is not None:
            weights = torch.from_numpy(weights).to(self.args.device).float()
        
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=weights)

    def set_trial(self, trial: Trial) -> None:
        """
        Args:
            trial (Trial): Optuna trial.
        Returns:
            None
        """
        self.trial = trial

    def set_optimizer(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        if self.args.optim == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "rmsprop":
            self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "radam":
            self.optimizer = optim.RAdam(
            self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "adamw":
            self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.args.optim_patience, factor=0.2)

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        try:
            model = prepare_model(self.args.backbone)
        except:
            model = prepare_model("r2plus1d_18")
        model = model.to(self.args.device)
        if self.args.device == "cuda":
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        self.model = model

    def save_params(self) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        self.storer.save_params(self.args)

    def show_model_info(self) -> None:
        """
        Show overview of model.
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        seqlen = int(self.args.freq * self.args.length)
        input_size = (self.args.bs, self.args.num_lead, seqlen)
        summary(self.model, input_size=input_size, device=self.args.device)

    def get_best_loss(self) -> float:
        """
        Args:
            None
        Returns:
            best_value (float):
        """
        return self.best

    def _train(self, iterator: Iterable):
        raise NotImplementedError

    def _evaluate(self, iterator: Iterable):
        raise NotImplementedError

    def run(self, train_loader: Iterable, valid_loader: Iterable):
        raise NotImplementedError