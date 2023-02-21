import os
from argparse import Namespace
from itertools import product
from typing import Dict, Optional

import yaml
import torch
import pandas as pd

from run_train import run_train
from codes.supports import utils

torch.backends.cudnn.deterministic = True

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

SEED = 1
MUST_KEYS = ("task", "search_space")

def run_gs(
    args: Namespace,
    gs_config: Dict,
    info_string: Optional[str]=None,
) -> str:
    """
    Args:
        args (Namespace):
        hps_name (str): Name of hyperparameter search target.
            must have keys listed in `MUST_KEYS`. 
        info_string (str):
    Returns:
        save_loc (str):
    """
    assert all (key in gs_config for key in MUST_KEYS)

    # Prepare save loc.
    tstamp = utils.get_timestamp()
    save_loc = os.path.join(
        config["gs_result_loc"], "grid_search", tstamp)
    os.makedirs(save_loc, exist_ok=True)

    # Storing setting.
    gs_result = []
    columns = list(gs_config["search_space"].keys()) + ["loss", "save_loc"]

    # Execute grid search.
    search_space_list = list(gs_config["search_space"].values())
    param_combinations = list(product(*search_space_list))
    for i, param_combination in enumerate(param_combinations):
        info_string = f"Gridsearch: {i+1} / {len(param_combinations)}"
        # Update grid search target parameters.
        param_vals = []
        for j, param_key in enumerate(gs_config["search_space"].keys()):
            args = utils.add_param(param_combination[j], param_key, args)
            param_vals.append(param_combination[j])

        # Run training and store result.
        best_loss, model_save_loc = run_train(args, info_string)
        gs_result.append(param_vals + [round(best_loss, 3), model_save_loc])

        # Store tmp result data.
        df_tmp = pd.DataFrame(gs_result, columns=columns)
        df_tmp.to_csv(save_loc + "/tmp.csv")

    # Save result.
    df_result = pd.DataFrame(gs_result, columns=columns)
    df_result.to_csv(save_loc + "/result.csv")
    return save_loc

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()
    
    # Prepare `Namespace`.
    with open(args.config) as f:
        params = yaml.safe_load(f)

    base_params = utils.prepare_params(params, SEED, args.device)

    # try:
    save_loc = run_gs(base_params, params)
    message = f"Grid search done. Result saved at `{save_loc}`@{base_params.host}."
    print(message)
    # except Exception as e:
    #     print(e)
