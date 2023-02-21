import os
from typing import Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import utils

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["tackle_detector"]

class DetectorDataPreparator:

    def __init__(self, seed: int=1) -> None:
        """
        Args:
            seed (int): Random seed value.
        Returns:
            None
        """

        self.train_v_ids, self.test_v_ids = self._prep_split_info()
        self.seed = seed
        
        df = pd.read_csv(config["src_video"]["video_id_csv"])
        self.video_id_dict = dict(zip(df.video_id, df.path))

    def _prep_split_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract array of video ids for train/test set
        based on common data split from dataset repository.

        Args:
            None
        Returns:
            train_v_ids (np.ndarray):
            test_v_ids (np.ndarray):
        """
        train_split_csv = os.path.join(
            config["datarepo"]["root"], 
            config["datarepo"]["train_test_split"]["train"]
        )
        test_split_csv = os.path.join(
            config["datarepo"]["root"], 
            config["datarepo"]["train_test_split"]["test"]
        )
        df_train = pd.read_csv(train_split_csv, index_col=0)
        df_test = pd.read_csv(test_split_csv, index_col=0)

        train_v_ids = df_train.loc[:, "video_id"].values
        test_v_ids = df_test.loc[:, "video_id"].values
        return train_v_ids, test_v_ids

    def _split_train_test(self, df_tackles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split list of tackles into train (+valid) set and test set.

        Args:
            df_tackles (pd.DataFrame):
        Returns:
            df_train (pd.DataFrame):
            df_test (pd.DataFrame):
        """
        v_ids = df_tackles.loc[:, "video_id"]
        is_train = np.isin(v_ids, self.train_v_ids)
        is_test = np.isin(v_ids, self.test_v_ids)

        df_train = df_tackles[is_train]
        df_test = df_tackles[is_test]
        return df_train, df_test

    def _split_train_valid(self, df: pd.DataFrame):
        """
        Split train set into train and valid.

        Args:
            df (pd.DataFrame):
        Returns:
            df_train (pd.DataFrame):
            df_valid (pd.DataFrame):
        """
        v_ids = df.loc[:, "video_id"].values
        train_v_ids, valid_v_ids = train_test_split(
            np.unique(v_ids), 
            test_size=config["valid_ratio"], 
            random_state=self.seed
        )

        df_train = df[np.isin(v_ids, train_v_ids)]
        df_valid = df[np.isin(v_ids, valid_v_ids)]
        return df_train, df_valid

    def _get_filepath(self, video_id: int, frame_id: int, label: str) -> Tuple[str, str]:
        """
        Args:
            video_id (int):
            frame_id (int):
            label (str):
        Returns:
            image_path (str):
            json_path (str):
        """
        if label == "nonHIA":
            data_dir = "nonHIAtackle"
        else:
            data_dir = "HIAtackle"

        # File path.
        jsonfile = f'video{video_id:06d}_frame{frame_id:07d}.json'
        imagefile = f'video{video_id:06d}_frame{frame_id:07d}.jpg'
        json_path = os.path.join(
            config["src_data_loc"], data_dir, "jsons", jsonfile)
        image_path = os.path.join(
            config["src_data_loc"], data_dir, "images", imagefile)

        return image_path, json_path

    def _save_processed_data(
        self,
        save_loc: str,
        bbox_text: str,
        image_path: str,
        class_idx: str
    ) -> None:
        """
        Save processed results.
            - save bbox as text file.
            - copy image file and save as jpg file.

        Args:
            save_loc (str):
            bbox_text (str):
            image_path (str):
            class_idx (str):
        Returns:
            None
        """
        with open(save_loc + f"/{self.data_idx:05d}.txt", "w") as f:
            f.write(f"{class_idx} {bbox_text}")

        # Copy image file.
        cp_image_path = f"{save_loc}/{self.data_idx:05d}.jpg"
        os.system(f"cp {image_path} {cp_image_path}")

        self.data_idx += 1
        self.datalist.append(cp_image_path)

    def _prep_random_frame(self, video_id: int, save_loc: str) -> None:
        """
        Args:
            video_id (int): 
        Returns:
            None
        """
        video_path = self.video_id_dict[video_id]
        video_loc = os.path.join(
            config["src_video"]["video_root"], video_path)
        
        cap = cv2.VideoCapture(video_loc)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        random_frame_idxs = [
            int(r_idx * total_frames) 
            for r_idx in np.random.rand(config["random_frame_ratio"])
        ]

        for idx in random_frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()
            if ret:
                # Save empty bbox file.
                with open(save_loc + f"/{self.data_idx:05d}.txt", "w") as f:
                    f.write(" ")

                # save frame.
                savename = f"{save_loc}/{self.data_idx:05d}.jpg"
                cv2.imwrite(savename, frame)

                self.data_idx += 1
                self.datalist.append(savename)

        cap.release()

    def _process_split(self, df: pd.DataFrame, datatype: str) -> None:
        """
        Args:
            df (pd.DataFrame):
            datatype (str): train, valid, test.
        Returns:
            None
        """
        print(f"Working on {datatype} data ...")

        num_ratio = f'random_ratio_{config["random_frame_ratio"]:02d}'
        save_loc = os.path.join(config["image_save_loc"], num_ratio, datatype)
        os.makedirs(save_loc, exist_ok=True)
        self.datalist = []
        self.data_idx = 1
        for _, row in tqdm(df.iterrows(), total=len(df)):
            label = row["label"]
            video_id, frame_id = row["video_id"], row["frame"]
            carrier_idx, tackler_idx = row["carrier"], row["tackler"]

            # Get path to image and json file.
            image_path, json_path =\
                 self._get_filepath(video_id, frame_id, label)

            # Label prep.
            class_idx = 0
            # Bbox prep.
            bbox = utils.prepare_bbox(json_path, carrier_idx, tackler_idx)
            scaled_bbox = utils.scale_bbox(bbox, image_path)
            bbox_text = " ".join(map(str, scaled_bbox))

            # Save data.
            self._save_processed_data(
                save_loc, bbox_text, image_path, class_idx)
            if config["random_frame_ratio"] > 0:
                self._prep_random_frame(video_id, save_loc)

        # Save list of path to image files.
        with open(save_loc + f"/{datatype}.txt", "w") as f:
            f.write("\n".join(self.datalist))

    def run(self) -> None:
        """
        Load `master.csv` and split

        Args:
            None
        Returns:
            None
        """
        tackle_listfile = os.path.join(
            config["csv_save_loc"], config["tackle_listfile"])
        df_tackles = pd.read_csv(tackle_listfile, index_col=0)
        df_train, df_test = self._split_train_test(df_tackles)
        df_train, df_valid = self._split_train_valid(df_train)

        self._process_split(df_train, "train")
        self._process_split(df_valid, "valid")
        self._process_split(df_test, "test")

        # Save csv.
        df_train = df_train.sort_values(by=["video_id"])
        df_train.to_csv(config["csv_save_loc"] + "/datasplit_csv_train.csv")
        df_valid = df_valid.sort_values(by=["video_id"])
        df_valid.to_csv(config["csv_save_loc"] + "/datasplit_csv_valid.csv")
        df_test = df_test.sort_values(by=["video_id"])
        df_test.to_csv(config["csv_save_loc"] + "/datasplit_csv_test.csv")

if __name__ == "__main__":

    preprator = DetectorDataPreparator()
    preprator.run()
