import os
from glob import glob
from typing import List, Tuple

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["ball_detector"]

class DetectorDataPreparator:

    def __init__(self, seed: int=1) -> None:
        """
        Args:
            seed (int): Random seed value.
        Returns:
            None
        """
        self.seed = seed
        self.save_loc = os.path.join(
            config["processed_data_loc"], "base")
        os.makedirs(self.save_loc, exist_ok=True)      

    def _split_data(
        self, 
        data_idxs: np.ndarray, 
        test_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split train set into train and valid.

        Args:
            data_idxs (np.ndarray):
            test_size (float): 
        Returns:
            train_idxs (np.ndarray): 
            test_idxs (np.ndarray): 
        """
        train_idxs, test_idxs = train_test_split(
            data_idxs, 
            test_size=test_size, 
            random_state=self.seed
        )
        return train_idxs, test_idxs

    def _save_data(
        self, 
        imgfiles: List, 
        target_idxs: np.ndarray, 
        datatype: str
    ) -> None: 
        """
        Args:
            imgfiles (np.ndarray): 
            targets (np.ndarray): 
            datatype (str): 
        Returns:
            None
        """
        save_loc = os.path.join(self.save_loc, datatype)
        os.makedirs(save_loc, exist_ok=True)
        print(f"Working on {datatype} data...")
        for target_idx in tqdm(target_idxs):
            imgfile = imgfiles[target_idx]
            basename, _ = os.path.splitext(os.path.basename(imgfile))

            savename_img = f"{save_loc}/{target_idx:04d}.jpg"
            command_img = f"cp {imgfile} {savename_img}"
            os.system(command_img)

            loadname_txt = f'{config["src_data_loc"]}/YOLO_annotations/{basename}.txt'
            savename_txt = f"{save_loc}/{target_idx:04d}.txt"
            command_txt = f"cp {loadname_txt} {savename_txt}"
            os.system(command_txt)

    def run(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        imgfiles = sorted(glob(config["src_data_loc"] + "/Images/*.jpg"))
        data_idxs = np.arange(len(imgfiles))
        train_idxs, test_idxs = self._split_data(
            data_idxs, test_size=1-config["train_size"])
        valid_idxs, test_idxs = self._split_data(
            test_idxs, test_size=0.5)

        # Store id table.
        img_id_table = list([[i+1, os.path.basename(f)] 
                              for i, f in enumerate(imgfiles)])
        csvname = os.path.join(self.save_loc, "img_id_table.csv")
        pd.DataFrame(img_id_table, columns=["id", "srcfile"]).to_csv(csvname)

        self._save_data(imgfiles, train_idxs, "train")
        self._save_data(imgfiles, valid_idxs, "valid")
        self._save_data(imgfiles, test_idxs, "test")


if __name__ == "__main__":

    preprator = DetectorDataPreparator()
    preprator.run()
