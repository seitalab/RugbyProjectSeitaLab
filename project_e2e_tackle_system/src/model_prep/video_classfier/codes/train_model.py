import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

from optuna.exceptions import TrialPruned

from codes.train_base import BaseTrainer
from codes.supports.monitor import Monitor, EarlyStopper

def print_result(monitor: Monitor, is_train: bool) -> None:
    """
    Args:
        monitor (Monitor): 
    Returns:
        None
    """
    loss = monitor.average_loss()
    score = monitor.macro_f1()
    recall_score = monitor.recall()
    auc_score = monitor.roc_auc_score()
    prec_score = monitor.precision()
    if is_train:
        mode = "Train"
    else: 
        mode = "Eval"
    result_string = (
        f'-> {mode} loss: {loss:.4f} '
        f'(F1: {score:.3f}, AUC-ROC: {auc_score:.3f}, '
        f'Recall: {recall_score:.3f}, '
        f'Precision: {prec_score:.3f})'
    )
    print(result_string)
    monitor.show_confusion_matrix()

class ModelTrainer(BaseTrainer):

    def _train(self, loader) -> Tuple[float, float]:
        """
        Run train mode iteration.
        Args:
            loader:
        Returns:
            score (float):
            loss (float): 
        """

        monitor = Monitor()
        self.model.train()

        # X, y, data_index; data_index not used during training.
        for X, y, _ in tqdm(loader):

            self.optimizer.zero_grad()
            X = X.to(self.args.device).float()
            y = y.to(self.args.device).float()
            y_pred = self.model(X)

            minibatch_loss = self.loss_fn(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss) * len(X))
            monitor.store_num_data(len(X))
            y_pred = torch.sigmoid(y_pred)
            monitor.store_result(y, y_pred)

        print_result(monitor, is_train=True)
        loss = monitor.average_loss()
        score = monitor.macro_f1()
        return score, loss
        
    def _evaluate(
        self, 
        loader, 
        monitor_id: bool=False
    ) -> Tuple[float, float, Optional[Tuple]]:
        """
        Args:
            loader :
            monitor_id (bool): If true track record of data id.
        Returns:
            score (float):
            loss (float): 
            records (Optional[Tuple]): 
        """
        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():

            for X, y, idx in tqdm(loader):
                X = X.to(self.args.device).float()
                y = y.to(self.args.device).float()

                y_pred = self.model(X)
                minibatch_loss = self.loss_fn(y_pred, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                y_pred = torch.sigmoid(y_pred)
                monitor.store_result(y, y_pred)

                if monitor_id:
                    monitor.store_data_idx(idx)

        print_result(monitor, is_train=False)
        loss = monitor.average_loss()
        score = monitor.macro_f1()
        if monitor_id:
            return score, loss, monitor.fetch_record()
        return score, loss
        
    def run(self, train_loader, valid_loader, mode="min") -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        assert mode in ["max", "min"]
        flip_val = -1 if mode == "max" else 1

        self.best = np.inf * flip_val # Sufficiently large or small
        if self.trial is None:
            early_stopper = EarlyStopper(
                mode=mode, patience=self.args.patience)

        for epoch in range(1, self.args.ep + 1):
            print("-" * 80)
            print(f"Epoch {epoch}")
            train_score, train_loss = self._train(train_loader)
            self.storer.store_epoch_result(
                epoch, train_loss, train_score, is_eval=False)

            if epoch % self.args.eval_every == 0:
                eval_score, eval_loss = self._evaluate(valid_loader)
                self.scheduler.step(eval_score)

                if mode == "max":
                    monitor_target = eval_score
                else:
                    monitor_target = eval_loss

                # Use pruning if hyperparameter search with optuna.
                # Use early stopping if not hyperparameter search (= trial is None).
                if self.trial is not None:
                    self.trial.report(monitor_target, epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
                else:
                    if early_stopper.stop_training(monitor_target):
                        break

                if self.info_string is not None:
                    print(self.info_string)

                if monitor_target * flip_val < self.best * flip_val:
                    print(f"Valid improved {self.best:.4f} -> {monitor_target:.4f}")
                    self.best = monitor_target
                    self.storer.save_model(self.model, monitor_target)
            self.storer.store_logs()