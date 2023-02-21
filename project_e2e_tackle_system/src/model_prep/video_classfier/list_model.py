import os
from glob import glob
from matplotlib.pyplot import show

import yaml

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["video_classifier"]

def show_model_locs() -> None:
    """
    Args:

    Returns:

    """
    model_locs = glob(config["save_root"] + "/*")
    for model_loc in model_locs:
        score_file = model_loc + "/best_score.txt"
        if not os.path.exists(score_file):
            continue

        score = open(score_file).read()
        print("-"*80)
        print(model_loc)
        print(score)

if __name__ == "__main__":

    show_model_locs()