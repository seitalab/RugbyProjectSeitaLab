import os
import pickle
from typing import Tuple

import yaml

from codes.eval_model import ModelEvaluator

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

def run_eval(eval_target: str, device: str) -> Tuple[float, float]:
    """
    Args:
        eval_target (str): Path to eval target.
        device (str):
    Returns:
        test_score (float): 
        test_loss (float): 
    """
    # Settings
    param_file = os.path.join(eval_target, "params.pkl")
    weightfile = os.path.join(eval_target, "net.pth")

    train_log_file = os.path.join(eval_target, "train_scores.json")
    eval_log_file = os.path.join(eval_target, "eval_scores.json")

    with open(param_file, "rb") as fp:
        params = pickle.load(fp)

    # Evaluator
    evaluator = ModelEvaluator(
        params, config["eval_result_loc"], device)
    evaluator.set_model()
    evaluator.set_lossfunc()
    evaluator.set_weight(weightfile)

    valid_loader, test_loader = evaluator.prepare_dataloader()

    # _ = evaluator.run(valid_loader, "valid")
    test_score, test_loss = evaluator.run(test_loader, "test")

    evaluator.dump_target(eval_target)
    evaluator.store_training_curve(train_log_file, "train", True)
    evaluator.store_training_curve(train_log_file, "train", False)
    evaluator.store_training_curve(eval_log_file, "eval", True)
    evaluator.store_training_curve(eval_log_file, "eval", False)

    print(f"Results saved at {evaluator.dump_loc}")

    return test_score, test_loss

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="v1")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    model_dictfile = "./resources/models.yaml"
    with open(model_dictfile) as f:
        model_dict = yaml.safe_load(f)
    target = model_dict[args.target]["ckptfile"]

    run_eval(target, args.device)