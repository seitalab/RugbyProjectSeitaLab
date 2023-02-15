import os
import json
import pickle

import torch
import torch.nn as nn

class Storer:

    def __init__(self, save_dir: str):
        """
        Args:
            save_dir (str): Path to save dir
        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir

        self.trains = {"loss": {}, "score": {}}
        self.evals = {"loss": {}, "score": {}}

    def save_params(self, params) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        savename = self.save_dir + "/params.pkl"
        with open(savename, "wb") as fp:
            pickle.dump(params, fp)

    def save_model(self, model: nn.Module, score: float) -> None:
        """
        Save current model (overwrite existing model).
        Args:
            model (nn.Module):
            score (float):
        Returns:
            None
        """
        savename = self.save_dir + "/net.pth"
        torch.save(model.state_dict(), savename)

        with open(self.save_dir + "/best_score.txt", "w") as f:
            f.write(f"{score:.5f}")

    def store_epoch_result(
        self, epoch: int, loss: float, score: float, is_eval: bool = False) -> None:
        """
        Args:
            epoch (int):
            score (float):
        Returns:
            None
        """
        if is_eval:
            self.evals["loss"][epoch] = loss
            self.evals["score"][epoch] = score
        else:
            self.trains["loss"][epoch] = loss
            self.trains["score"][epoch] = score

    def store_logs(self):
        """
        Args:
            None
        Returns:
            None
        """

        with open(self.save_dir + "/train_scores.json", "w") as ft:
            json.dump(self.trains, ft, indent=4)

        with open(self.save_dir + "/eval_scores.json", "w") as fe:
            json.dump(self.evals, fe, indent=4)