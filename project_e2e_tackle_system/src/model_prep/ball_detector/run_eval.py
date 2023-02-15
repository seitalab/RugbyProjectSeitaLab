import os
from glob import glob
from typing import List, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

import utils

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["ball_detector"]["detector_train"]

class ModelEvaluator:

    def __init__(self, eval_setting: str, mode:str, config_file: str, checkpoint_file: str, device: str) -> None:
        """
        Args:
            eval_setting (str):
            mode (str)
            config_file (str):
            checkpoint_file (str): Path to checkpoint_file.
        Returns:
            None
        """
        self.savebase = os.path.join(
            config["result_eval"]["result_save_loc"], mode, eval_setting)
        self.data_loc = os.path.join(config["image_data_loc"], mode)

        self._prep_model(config_file, checkpoint_file, device)

    def _prep_model(self, config_file: str, checkpoint_file: str, device: str) -> None:

        self.model = init_detector(config_file, checkpoint_file, device=device)

    def _draw_gt_bbox(self, img_file: str, img_data: np.ndarray) -> Tuple[np.ndarray, Tuple, int]:
        """
        Args:
            img_file (str):
            img_data (np.ndarray):
        Returns:
            img_data (np.ndarray):
            bbox (Tuple): Tuple of left_top and right_bottom of x, y coordinates.
            class_idx (int):
        """
        img_h, img_w, _ = img_data.shape
        _, extension = os.path.splitext(img_file)

        bboxfile = img_file.replace(extension, ".txt") # img and bbox txt must be at same dir.
        bbox = utils.load_bbox(bboxfile, img_w, img_h)
        label_idx = utils.extract_label(bboxfile)
        if bbox[0] is not None:
            img_data = utils.draw_bbox(img_data, bbox, label_idx, is_gt=True)
        return img_data, bbox, label_idx

    def _draw_pred_bbox(self, bboxes: List, img_data: np.ndarray) -> Tuple[Tuple, List, List]:
        """
        Args:
            bboxes (List):
            img_data (np.ndarray):
        Returns:
            img_data (np.ndarray):
            valid_bboxes (List):
            top_bboxes (List):
        """
        valid_bboxes = []
        for bbox_ in bboxes:
            for bbox in bbox_:
                conf = bbox[4]
                if conf > config["result_eval"]["threshold"]:
                    bbox = (bbox[:2].astype(int), bbox[2:4].astype(int))
                    img_data = utils.draw_bbox(img_data, bbox)
                    valid_bboxes.append(bbox)

        top_bboxes = []
        for bbox_ in bboxes:
            confs = np.array([bbox[4] for bbox in bbox_])
            if confs.size != 0:
                bbox = bbox_[np.argmax(confs)]
                bbox = (bbox[:2].astype(int), bbox[2:4].astype(int))
                top_bboxes.append(bbox)

        return img_data, valid_bboxes, top_bboxes

    def _detect(self, img_file: str) -> List:
        """
        Args:
            img_file (str):
        Returns:
            bboxes
        """
        # Returns [[left_top_x, left_top_y, right_bot_x, right_bot_y, conf]]
        bboxes = inference_detector(self.model, img_file)
        return bboxes

    def process_image(self, img_file: str) -> Tuple:
        """
        Args:
            img_file (str):
        Returns:
            bbox (Tuple):
                gt_bbox (np.ndarray): 
                valid_bboxes (np.ndarray)
            valid_bbox_scores (np.ndarray):
            top_bbox_scores (np.ndarray):
        """
        img_data = cv2.imread(img_file)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        img_data, gt_bbox, _ = self._draw_gt_bbox(img_file, img_data)
        pred_bboxes = self._detect(img_file)
        img_data, valid_bboxes, top_bboxes =\
            self._draw_pred_bbox(pred_bboxes, img_data)

        valid_bbox_scores = utils.calc_score(gt_bbox, valid_bboxes)
        top_bbox_scores = utils.calc_score(gt_bbox, top_bboxes)

        img_data = cv2.resize(img_data, (480, 360))
        savename = img_file.replace(self.data_loc, self.savebase)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(savename, img_data)

        return (gt_bbox, valid_bboxes), valid_bbox_scores, top_bbox_scores

    def run(self, datatype: str):
        """
        Args:
            datatype (str): train/valid/test
        Returns:

        """
        print(f"Working on {datatype} dataset ...")
        save_loc = self.savebase + f"/{datatype}"
        os.makedirs(save_loc, exist_ok=True)

        img_files = sorted(glob(self.data_loc + f"/{datatype}/*.jpg"))
        df_result = pd.DataFrame(columns=config["result_eval"]["columns"])

        for img_file in tqdm(img_files):
            bbox, valid_bbox_scores, top_bbox_scores =\
                self.process_image(img_file)
            basename = os.path.basename(img_file)
            
            if len(valid_bbox_scores) == 0:
                best_iou = 0
                avg_iou = 0
                top_bbox_iou = 0
            else:
                best_iou = np.round(np.max(valid_bbox_scores), 3)
                avg_iou = np.round(np.mean(valid_bbox_scores), 3)
                top_bbox_iou = np.round(np.max(top_bbox_scores), 3)
            
            num_gt_bbox = int(bbox[0][0] is not None)
            if bbox[0][0] is not None:
                num_detected = len(bbox[1]) * int(top_bbox_iou > 0)
            else:
                num_detected = len(bbox[1])

            row = {
                "filename": basename,
                "gt_bbox": num_gt_bbox, 
                "num_bbox": num_detected,
                "best_iou": best_iou,
                "avg_iou": avg_iou,
                "top_bbox_iou": top_bbox_iou,
            }
            df_result = df_result.append(row, ignore_index=True)
        savename = self.savebase + f"/summary_{datatype}.csv"
        df_result.to_csv(savename)

        report = utils.calc_detector_score(df_result)
        print(report)
        with open(self.savebase + f"/report_{datatype}.txt", "w") as f:
            f.write(report)

if __name__ == "__main__":
    from argparse import ArgumentParser

    model_list_file = "models.yaml"
    with open(model_list_file) as f:
        model_dict = yaml.safe_load(f)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="retina")
    parser.add_argument('--mode', type=str, default="base")
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    config_file = model_dict[args.model]["cfgfile"]
    checkpoint_file = model_dict[args.model]["ckptfile"]

    evaluator = ModelEvaluator(
        args.model, args.mode, config_file, checkpoint_file, args.device)
    # evaluator.run("valid")
    evaluator.run("test")
