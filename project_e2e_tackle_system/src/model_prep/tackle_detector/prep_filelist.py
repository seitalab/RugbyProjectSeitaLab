import os

import yaml
import numpy as np
import pandas as pd

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["tackle_detector"]

def main() -> None:
    """
    Args:
        None
    Returns:
        None
    """
    HIAtackle_path = os.path.join(
        config["datarepo"]["root"], 
        config["datarepo"]["event_list_csv"]["hia_tackle"]
    )
    nonHIAtackle_path = os.path.join(
        config["datarepo"]["root"], 
        config["datarepo"]["event_list_csv"]["nonhia_tackle"]
    )
    # Read csv.
    HIAtackle = pd.read_csv(HIAtackle_path)
    nonHIAtackle = pd.read_csv(nonHIAtackle_path)

    # carrier,tackler選定成功サンプルのみ抽出
    HIAtackle = HIAtackle[HIAtackle.tackler != 0]
    nonHIAtackle = nonHIAtackle[nonHIAtackle.tackler != 0]

    # ラベル付与して統合
    nonHIAtackle["label"] = "nonHIA"
    master = pd.concat([HIAtackle, nonHIAtackle]).reset_index()

    # IDを付与
    master["tackle_id"] = np.arange(1, master.shape[0] + 1)
    master = master.loc[
        :,
        [
            "tackle_id",
            "video_id",
            "frame",
            "tackler",
            "carrier",
            "label",
        ],
    ]

    # Save.
    os.makedirs(config["csv_save_loc"], exist_ok=True)
    savename = os.path.join(
        config["csv_save_loc"], config["tackle_listfile"])
    master.to_csv((savename), index=False)


if __name__ == "__main__":

    main()
