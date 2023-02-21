import os
from typing import List

import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import utils
from prep_data_with_random import DetectorDataPreparator

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
add_data_cfg = config["additional_data_prep"]
config = config["tackle_detector"]

def load_bbox(bboxfile: str, image_w: int, image_h: int) -> List:
    """
    Assuming input bbox as `[tl_x, tl_y, br_x, br_y, conf]` format.
    Returns [center_x, center_y, w, h].

    Args:
        bboxfile (str):
        image_w (int):
        image_h (int):
    Returns:
        bbox (List): 
    """
    bbox = open(bboxfile, "r").read() # [tl_x, tl_y, br_x, br_y, conf]
    bbox = [float(v) for v in bbox.split(" ")[:4]]
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # scale by image size.
    center_x = int(center_x / image_w)
    center_y = int(center_y / image_h)
    width = int(width / image_w)
    height = int(height / image_h)
    bbox = [center_x, center_y, width, height]
    return bbox

class LabeledDataPreparator(DetectorDataPreparator):

    def _add_manually_labeled(self, save_loc: str) -> None:
        """
        Args:
            save_loc (str): 
        Returns:
            None
        """
        base_loc = os.path.join(
            add_data_cfg["image_save_loc"], 
            f'nframe-{add_data_cfg["num_frame_per_video"]:06d}'
        )
        label_csv = os.path.join(
            base_loc, 
            add_data_cfg["manual_label_csv"]
        )
        df_label = pd.read_csv(label_csv).fillna(0)
        for _, row in tqdm(df_label.iterrows(), total=len(df_label)):
            src_image = os.path.join(base_loc, "raw", row["filename"])
            image = cv2.imread(src_image)
            dst_image = save_loc + f"/{self.data_idx:05d}.jpg"
            cv2.imwrite(dst_image, image)

            if row["label"] == 0:
                # Save empty bbox label.
                with open(save_loc + f"/{self.data_idx:05d}.txt", "w") as f:
                    f.write(" ")
            else:
                txt_file = row["filename"].replace(".jpg", ".txt")
                src_label = os.path.join(base_loc, "label", txt_file)
                image_h, image_w, _ = image.shape
                scaled_bbox = load_bbox(src_label, image_w, image_h) # out: [center_x, center_y, w, h]
                scaled_bbox = [0] + scaled_bbox # -> [class_label=0, center_x, center_y, w, h]
                bbox_text = " ".join(map(str, scaled_bbox))
                dst_label = save_loc + f"/{self.data_idx:05d}.txt"
                with open(dst_label, "w") as f:
                    f.write(bbox_text)

            # Update data idx and datalist.
            self.data_idx += 1
            self.datalist.append(dst_image)

    def _process_split(self, df: pd.DataFrame, datatype: str) -> None:
        """
        Args:
            df (pd.DataFrame):
            datatype (str): train, valid, test.
        Returns:
            None
        """
        print(f"Working on {datatype} data ...")

        num_ratio = f'add_nframe-{add_data_cfg["num_frame_per_video"]:06d}'
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
        
        if datatype == "train":
            self._add_manually_labeled(save_loc)

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

    preprator = LabeledDataPreparator()
    preprator.run()
