import os
from glob import glob
from typing import List, Tuple

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["eval_component"]

class Evaluator:

    def __init__(self):
        pass

    def _load_gt(
        self, 
        video_file: str, 
    ) -> np.array:
        """
        Args:
            video_file (str): 
        Returns:
            gt_labels (np.ndarray): 
        """
        csvfile = os.path.join(
            config["manual_label_csv"], 
            os.path.basename(video_file)[:-4]+".csv"
        )

        df = pd.read_csv(csvfile)
        return df.loc[:, "is_tackle"].values

    def _load_pred(
        self, 
        video_file: str, 
        video_classifier: str
    ) -> np.ndarray:
        """
        Args:

        Returns:
            pred_frame_idx (np.ndarray): Array of frame index predicted as tackle frame by given classifier.
        """
        basename, _ = os.path.splitext(os.path.basename(video_file))
        load_loc = os.path.join(
            config["data_root"], 
            config["eval_clf"], 
            video_classifier, 
        )
        csv_file = load_loc + f"/{basename}.csv"
        df = pd.read_csv(csv_file)
        return df.loc[:, "tackle_confidence"].values

    def evaluate(
        self, 
        video_files: List, 
        video_classifier: str, 
        thres: float=0.5
    ):
        """
        Args:
            video_files (List): 
            video_classifier (str): 
        Returns:
            None
        """
        scores = []
        for video_file in tqdm(video_files):
            gt_label = self._load_gt(video_file)
            pred_label = self._load_pred(video_file, video_classifier)

            if gt_label.sum() == 0:
                continue

            pred_label = (pred_label > thres).astype(int)

            f1_mic = f1_score(gt_label, pred_label, average="micro")
            f1_mac = f1_score(gt_label, pred_label, average="macro")
            f1_pos = f1_score(gt_label, pred_label, average="binary") # score only for pos-label
            rec = recall_score(gt_label, pred_label)
            prec = precision_score(gt_label, pred_label)
            scores.append([f1_pos, f1_mic, f1_mac, rec, prec])

        scores = np.array(scores)
        mean_score = scores.mean(axis=0)
        print(np.round(mean_score, 3))

if __name__ == "__main__":

    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")

    evaluator = Evaluator()
    video_classifier = "no_clf"
    print(video_classifier)
    evaluator.evaluate(video_files, video_classifier)

    video_classifier = "mc3_r2_search03"
    print(video_classifier)
    evaluator.evaluate(video_files, video_classifier, 0.05)
    evaluator.evaluate(video_files, video_classifier, 0.1)
    evaluator.evaluate(video_files, video_classifier, 0.25)
    evaluator.evaluate(video_files, video_classifier, 0.5)

    video_classifier = "r2p_r2_search04"
    print(video_classifier)
    evaluator.evaluate(video_files, video_classifier, 0.05)
    evaluator.evaluate(video_files, video_classifier, 0.1)
    evaluator.evaluate(video_files, video_classifier, 0.25)
    evaluator.evaluate(video_files, video_classifier, 0.5)

    video_classifier = "r3d_r2_search05"
    print(video_classifier)
    evaluator.evaluate(video_files, video_classifier, 0.05)
    evaluator.evaluate(video_files, video_classifier, 0.1)
    evaluator.evaluate(video_files, video_classifier, 0.25)
    evaluator.evaluate(video_files, video_classifier, 0.5)

