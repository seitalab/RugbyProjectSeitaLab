import os
import pickle
from glob import glob
from typing import Dict, List, Tuple

import cv2
import yaml
import numpy as np
from tqdm import tqdm
from mmpose.apis import (
    init_pose_model, inference_bottom_up_pose_model, vis_pose_result
)

import utils

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["hia_detector"]

class PoseDetector:

    def __init__(
        self, 
        video_classifier: str,
        pose_detector: str, 
        config_file: str, 
        checkpoint_file: str, 
        device: str
    ) -> None:
        """
        Args:
            modelname (str): 
            config_file (str): 
            checkpoint_file (str): 
            device (str): 
        Returns:
            None
        """
        
        self.model = init_pose_model(config_file, checkpoint_file, device=device)
        self.pose_detector = pose_detector
        self.video_classifier = video_classifier

    def _detect(self, frame: np.ndarray) -> List: 
        """
        Apply detection to single frame.

        Args:
            frame (np.ndarray): 
        Returns:
            keypoints (List): List of keypoint dicts 
                eg. [{"bbox": array(lt_x, lt_y, rb_x, rb_y), "keypoints:[array(x, y, conf)x133]}, ..]
        """
        # `inference_bottom_..` returns Tuple, we only use first element.
        # https://mmpose.readthedocs.io/en/latest/api.html
        keypoints = inference_bottom_up_pose_model(self.model, frame)[0]
        return keypoints

    def _apply(self, frame_block_locs: List) -> Tuple[Dict, Dict]:
        """
        Args:
            frame_block_locs (List): List of frame blocks 
                [vidXXX_frameYYYYYY/frameZZZZZ, vidXXX_frameYYYYYY/frameZZZZZZ, ...]
        Returns:
            keypoint_dict 
            images 
        """

        keypoint_dict, images = {}, {}
        for frame_block_loc in frame_block_locs:
            # extract frame id.
            frame_id = os.path.basename(frame_block_loc)

            # fetch last frame file.
            frame_file = sorted(glob(frame_block_loc + "/*.jpg"))[-1]
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply tackle detector to frame.
            keypoints = self._detect(frame)
            keypoints = list(
                filter(lambda x: x["score"] > config["pose_threshold"], keypoints))
            img = vis_pose_result(self.model, frame, keypoints)

            images[frame_id] = img
            keypoint_dict[frame_id] = keypoints
        
        return keypoint_dict, images

    def detect_pose(self, video_file: str) -> None:
        """
        Args:
            video_file (str): 
            tackle_detector (str): 
        Returns:
            None
        """
        # Prep save loc.
        save_loc, load_loc = utils.prepare_dirpath(
            video_file=video_file,
            video_classifier=self.video_classifier,
            tackle_detector=None,
            pose_detector=self.pose_detector,
            ball_detector=None,
            classifier_name=None,
            mode="apply_pose_detector",
            config=config,
        )
        os.makedirs(save_loc, exist_ok=True)
        frame_block_locs = glob(load_loc + "/frame*")

        # Apply model
        keypoint_dict, images = self._apply(frame_block_locs)

        # Save.
        savename_kp = save_loc + "/keypoint_dict.pkl"
        with open(savename_kp, "wb") as fp:
            pickle.dump(keypoint_dict, fp)

        print("Saving frames ...")
        for frame_idx in tqdm(images.keys()):
            frame = images[frame_idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            savename = save_loc + f"/{frame_idx}.jpg"
            cv2.imwrite(savename, frame)
        print("Done")

if __name__ == "__main__":
    from glob import glob
    from argparse import ArgumentParser

    param_file = "./resource/model_info.yaml"
    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"

    parser = ArgumentParser()
    parser.add_argument('--pose-detector', type=str, default="hrnet")
    parser.add_argument('--clf', type=str, default="mc3_v1")
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    with open(param_file) as f:
        params = yaml.safe_load(f)

    config_file = params["pose_detect"][args.pose_detector]["config_file"]
    ckpt_file = params["pose_detect"][args.pose_detector]["checkpoint_file"]

    detector = PoseDetector(
        args.clf, args.pose_detector, config_file, ckpt_file, args.device)
    video_files = glob(video_loc + "/*.mp4")
    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        detector.detect_pose(video_file)