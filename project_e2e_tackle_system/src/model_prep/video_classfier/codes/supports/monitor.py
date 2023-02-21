from typing import Tuple

import numpy as np
from torch import Tensor
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, 
    confusion_matrix, accuracy_score, precision_score
)

class Monitor:

    def __init__(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        self.num_data = 0
        self.total_loss = 0
        self.ytrue_record = None
        self.ypred_record = None
        self.data_idx_record = []

    def _concat_array(self, record, new_data: np.array) -> np.ndarray:
        """
        Args:
            record ()
            new_data (np.ndarray):
        Returns:
            concat_data (np.ndarray):
        """
        if record is None:
            return new_data
        else:
            return np.concatenate([record, new_data])

    def store_loss(self, loss: float) -> None:
        """
        Args:
            loss (float): Mini batch loss value.
        Returns:
            None
        """
        self.total_loss += loss

    def store_num_data(self, num_data: int) -> None:
        """
        Args:
            num_data (int): Number of data in mini batch.
        Returns:
            None
        """
        self.num_data += num_data

    def store_result(self, y_trues: Tensor, y_preds: Tensor) -> None:
        """
        Args:
            y_trues (Tensor):
            y_preds (Tensor): Array with 0 - 1 values.
        Returns:
            None
        """
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        self.ytrue_record = self._concat_array(self.ytrue_record, y_trues)
        self.ypred_record = self._concat_array(self.ypred_record, y_preds)
        assert(len(self.ytrue_record) == len(self.ypred_record))

    def store_data_idx(self, data_indices: Tuple) -> None:
        """
        Args:
            data_indices (Tuple): 
        Returns:
            None
        """
        self.data_idx_record = self.data_idx_record + list(data_indices)
        assert(len(self.ytrue_record) == len(self.data_idx_record))

    def average_loss(self) -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_loss / self.num_data

    def macro_f1(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): Macro averaged F1 score.
        """
        y_preds = (self.ypred_record > 0.5).astype(int)
        score = f1_score(self.ytrue_record, y_preds, average='macro')
        return score

    def recall(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): Recall score.
        """
        y_preds = (self.ypred_record > 0.5).astype(int)
        score = recall_score(self.ytrue_record, y_preds)
        return score

    def precision(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):  Precision score.
        """
        y_preds = (self.ypred_record > 0.5).astype(int)
        score = precision_score(self.ytrue_record, y_preds)
        return score

    def accuracy(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """            
        y_preds = np.argmax(self.ypred_record, axis=1)
        score = accuracy_score(self.ytrue_record, y_preds)
        return score

    def roc_auc_score(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): AUC-ROC score.
        """
        try:
            score = roc_auc_score(self.ytrue_record, self.ypred_record)
        except:
            score = 0
        return score

    def show_confusion_matrix(self) -> None:
        """
        Args:
            is_multilabel (bool): 
        Returns:
            None
        """
        y_preds = (self.ypred_record > 0.5).astype(int)
        conf_matrix = confusion_matrix(self.ytrue_record, y_preds)
        print("Confusion Matrix")
        print(conf_matrix)

    def fetch_record(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            None
        Returns:
            y_true
            y_pred
            data_idx
        """
        data_idx = np.array(self.data_idx_record)
        return self.ytrue_record, self.ypred_record, data_idx



class EarlyStopper(object):

    def __init__(self, mode: str, patience: int):
        """
        Args:
            mode (str): max or min
            patience (int):
        Returns:
            None
        """
        assert (mode in ["max", "min"])
        self.mode = mode

        self.patience = patience
        self.num_bad_count = 0

        if mode == "max":
            self.best = -1 * np.inf
        else:
            self.best = np.inf

    def stop_training(self, metric: float):
        """
        Args:
            metric (float):
        Returns:
            stop_train (bool):
        """
        if self.mode == "max":

            if metric <= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        else:

            if metric >= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        if self.num_bad_count > self.patience:
            stop_train = True
            print("Early stopping applied, stop training")
        else:
            stop_train = False
            print(f"Patience: {self.num_bad_count} / {self.patience}")
        return stop_train