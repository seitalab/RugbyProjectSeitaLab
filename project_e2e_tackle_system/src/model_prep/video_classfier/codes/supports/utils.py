from datetime import datetime
from typing import Dict, Union, Optional
from argparse import Namespace

import socket

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

ParamsStringIgnore = ["device"]

def prepare_param_string(args_dict: Dict) -> str:
    """
    Args:
        args_dict (Dict): Dictionary of arguments.
    Returns:
        param_string (str):
    """
    param_string = ""
    for key, value in args_dict.items():
        if key in ParamsStringIgnore:
            continue
        param_string += f"{key}-{value}_"
    param_string = param_string[:-1]  # Remove last '_'
    return param_string

def prepare_params(
    params: Dict, 
    seed: int, 
    device: str
) -> Namespace:
    """
    Args:
        base_params (Dict): 
        seed (int): 
        device (str): 
    Returns:
        base_params (Namespace): 
    """
    base_params = Namespace(**params["base"])
    base_params.seed = seed
    base_params.host = socket.gethostname()
    base_params.device = device

    # # Merge parameters
    # base_params = merge_params(
    #     params["model"]["backbone"]["params"], base_params)
    # base_params.backbone = params["model"]["backbone"]["type"]

    return base_params

def merge_params(params: Optional[Dict], base_params: Namespace) -> Namespace:
    """
    Args:
        params (Dict): 
        base_params (Namespace): 
    Returns:
        merged_params (Namespace): 
    """
    if params is None:
        return base_params

    for key, values in vars(base_params).items():
        if key not in params:
            params[key] = values
    merged_params = Namespace(**params)
    return merged_params

def add_param(
    param_value: Union[int, float], 
    param_key: str, 
    base_params: Namespace
) -> Namespace:
    """
    Args:
        param_value (Union[int, float]): 
        param_key (str): 
        base_params (Namespace):
    Returns:
        params (Namespace)
    """
    params = vars(base_params)
    params.update({param_key: param_value})
    params = Namespace(**params)
    return params
