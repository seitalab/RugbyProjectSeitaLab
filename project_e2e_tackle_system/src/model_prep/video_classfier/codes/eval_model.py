import os
import json
from re import S
from typing import Iterable, Tuple
from argparse import Namespace

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from codes.train_model import ModelTrainer
from codes.data.dataloader import prepare_dataloader
from codes.supports.utils import get_timestamp
sns.set()

class ModelEvaluator(ModelTrainer):

    def __init__(self, args: Namespace, dump_loc: str, device: str) -> None:
        """
        Args:
            args (Namespace):
            dump_loc (str):
            device (str):
        Returns:
            None
        """
        self.args = args
        self.args.device = device

        self.device = device
        self.model = None

        timestamp = get_timestamp()
        self.dump_loc = os.path.join(dump_loc, timestamp)

        os.makedirs(self.dump_loc, exist_ok=True)

    def set_weight(self, weight_file: str):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def prepare_dataloader(self) -> Tuple[Iterable, Iterable]:
        """
        Args:
            None
        Returns:
            train_loader (Iterable):
            valid_loader (Iterable):
        """

        # Prepare dataloader.
        _, valid_loader, test_loader = prepare_dataloader(self.args)
        return valid_loader, test_loader

    def run(self, loader, datatype: str) -> Tuple[float, float]:
        """
        Args:
            loader
            datatype (str): 
        Returns:
            eval_score (float):
            eval_loss (float):
        """
        eval_score, eval_loss, records =\
            self._evaluate(loader, monitor_id=True)

        # Store prediction result for each data.
        trues, preds, idxs = records
        sorted_order = np.argsort(idxs)
        preds = (preds[sorted_order] > 0.5).astype(int)
        trues = trues[sorted_order].astype(int)
        stacked = np.stack([idxs[sorted_order], trues, preds])

        savename = self.dump_loc + f"/records_{datatype}.csv"
        df = pd.DataFrame(stacked.T, columns=["file_id", "label", "prediction"])
        df.to_csv(savename, index=False)

        return eval_score, eval_loss

    def store_training_curve(
        self, 
        logfile: str, 
        datatype: str, 
        is_score: bool=False
    ) -> None:
        """
        Args:
            logfile (str):
            datatype (str):
            is_score (bool): 
        Returns:
            None
        """
        with open(logfile, "r") as f:
            logdata = json.load(f)
        if is_score:
            target = "score"
        else:
            target = "loss"

        savename = self.dump_loc + f"/{datatype}_log_{target}.png"
        plt.plot(logdata[target].keys(), logdata[target].values())
        plt.xlabel("Epochs")
        plt.xticks(rotation='vertical')
        plt.ylabel(target)
        plt.savefig(savename)
        plt.close()

    def dump_target(self, eval_target: str):
        """
        Args:
            eval_target (str):
        Returns:
            None
        """
        with open(self.dump_loc + "/eval_target.txt", "w") as f:
            f.write(eval_target)