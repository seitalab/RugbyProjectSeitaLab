import os
from typing import Optional, Tuple
from argparse import Namespace

import yaml
import torch
import numpy as np
from optuna.trial import Trial

from codes.supports import utils
from codes.train_model import ModelTrainer

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

torch.backends.cudnn.deterministic = True

def calc_class_weight(labels: np.ndarray) -> np.ndarray:
    """
    Calculate class weight.
    Args:
        labels (np.ndarray): Label data array of shape [num_sample]
    Returns:
        class_weight (np.ndarray): Array of shape [num_classes].
    """
    num_samples = labels.shape[0]

    positive_per_class = labels.sum(axis=0)
    negative_per_class = num_samples - positive_per_class

    class_weight = negative_per_class / positive_per_class

    return np.array(class_weight)

def run_train(
    args: Namespace, 
    info_string: Optional[str]=None,
    trial: Optional[Trial]=None
) -> Tuple[float, str]:
    """
    Execute train code for tackle classifier
    Args:
        args (Namespace): Namespace for parameters used.
        info_string (Optional[str]): 
        trial (Optional[Trial]): Only used for hyper parameter optimization.
    Returns:
        best_val_loss (float): 
        save_dir (str):
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare result storing directories
    timestamp = utils.get_timestamp()
    save_setting = f"{timestamp}-{args.host}"
    save_dir = os.path.join(config["save_root"], save_setting)

    # Trainer prep
    trainer = ModelTrainer(args, save_dir, info_string)

    trainer.set_model()
    # trainer.show_model_info()

    print("Preparing dataloader ...")
    train_loader, valid_loader = trainer.prepare_dataloader()

    weight = calc_class_weight(train_loader.dataset.label)

    trainer.set_lossfunc(weight)
    trainer.set_optimizer()
    trainer.set_trial(trial)
    trainer.save_params()

    print("Starting training ...")
    trainer.run(train_loader, valid_loader)
    best_val_loss = trainer.get_best_loss()

    del trainer

    # Return best validation loss when executing hyperparameter search.
    return best_val_loss, save_dir

if __name__ == "__main__":
    from argparse import ArgumentParser

    import yaml

    parser = ArgumentParser()
    parser.add_argument(
        '--config', 
        help='path to config file', 
        default="./resources/trial.yaml"
    )
    parser.add_argument('--device', default="cuda:1")
    args = parser.parse_args()

    # Prepare `Namespace`.
    with open(args.config) as f:
        params = yaml.safe_load(f)

    base_params = utils.prepare_params(params, params["seed"], args.device)
    _, save_dir = run_train(base_params)
    print(save_dir)