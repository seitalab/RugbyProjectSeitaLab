import os
import sys
import pickle
from glob import glob
from typing import List

import cv2
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append("./CenterTrack/src")
import _init_paths
from opts import opts
from detector import Detector

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
ct_loc = config["center_track"]
config = config["hia_detector"]

limb_pair = [
    (0, 2), (2, 4), 
    (0, 1), (1, 3), 
    (5, 6), (6, 8), (8, 10), 
    (5, 7), (7, 9), 
    (12, 11), (6, 12), (5, 11),
    (12, 14), (14, 16), 
    (11, 13), (13, 15),
]

class CenterTrackDetector:

    def __init__(self, opt, video_clf: str):

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.debug = max(opt.debug, 1)
        self.detector = Detector(opt)
        self.thres = config["pose_threshold"]

        self.video_classifier = video_clf

    def _draw(self, img, det):
        # (34, ) -> (17, 2)
        for dot in det["hps"].reshape(-1, 2):
            cv2.circle(
                img, 
                center=dot.astype(int), 
                radius=5, 
                color=(20,20,20),
                thickness=-1
            )
        dots = det["hps"].reshape(-1, 2)
        for limb_src, limb_dst in limb_pair:
            cv2.line(
                img,
                dots[limb_src].astype(int),
                dots[limb_dst].astype(int),
                color=(255, 255, 255),
                thickness=2,
            )
        return img

    def _viz_with_index(self, frame: np.ndarray, keypoints: List) -> np.ndarray:
        """
        Args:
            frame (np.ndarray): 
            keypoints (List): List of dictionaries.
        Returns:
            frame (np.ndarray): 
        """
        # Do nothing if detection is empty.
        if len(keypoints) == 0:
            return frame
        
        for j in range(len(keypoints)):
            # Exclude pose with low score.
            if keypoints[j]["score"] < config["pose_threshold"]:
                continue

            # right_shoulder, left_shoulder, right_hip, left_hip
            body_points = np.array([
                keypoints[j]["keypoints"][i][:2] for i in [5, 6, 11, 12]
            ])
            # Exclude if any point is negative.
            body_points = body_points[(body_points > 0).all(axis=1)]
            center = body_points.mean(axis=0).astype(int)

            try:
                cv2.rectangle(
                    frame, 
                    pt1=center - (0, 30), 
                    pt2=center + (40, 5), 
                    color=(255, 255, 255), 
                    thickness=-1
                )
                cv2.putText(
                    frame, 
                    text=f"{j}", 
                    org=center, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(10, 10, 10),
                    thickness=2,
                    lineType=cv2.LINE_4
                )
            except:
                print("----")
                print(body_points)
                print(center)
        return frame

    def detect(self, imagefile: str, draw_on_image: bool=True):
        """
        Args:
            imagefile (str): 
        Returns:
            None
        """
        img = cv2.imread(imagefile)
        ret = self.detector.run(img)

        ret_store = []
        for det in ret["results"]:
            ret_store.append(
                {
                    "score": det["score"],
                    "keypoints": det["hps"].reshape(-1, 2)
                }
            )
            if det["score"] < self.thres:
                continue
            
            if draw_on_image:
                self._draw(img, det)
        return ret_store, img

    def apply_to_video(self, video_file: str):
        """
        Args:
            video_file (str): 
            tackle_detector (str): 
        Returns:
            None
        """
        # Prep save loc.
        basename, _ = os.path.splitext(os.path.basename(video_file))
        save_loc = os.path.join(
            ct_loc["pose_frame_loc"],
            basename
        )
        load_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            self.video_classifier, 
            basename
        )
        os.makedirs(save_loc, exist_ok=True)

        keypoint_dict, images = {}, {}
        frame_block_locs = glob(load_loc + "/frame*")
        for frame_block_loc in tqdm(frame_block_locs):
            # extract frame id.
            frame_id = os.path.basename(frame_block_loc)

            # fetch last frame file.
            frame_file = sorted(glob(frame_block_loc + "/*.jpg"))[-1]
            keypoints, img = self.detect(frame_file, True)
            keypoints = list(
                filter(lambda x: x["score"] > config["pose_threshold"], keypoints))
            img = self._viz_with_index(img, keypoints)

            images[frame_id] = img
            keypoint_dict[frame_id] = keypoints

        # Save.
        savename_kp = save_loc + "/keypoint_dict.pkl"
        with open(savename_kp, "wb") as fp:
            pickle.dump(keypoint_dict, fp)

        print("Saving frames ...")
        for frame_idx in tqdm(images.keys()):
            frame = images[frame_idx]

            savename = save_loc + f"/{frame_idx}.jpg"
            cv2.imwrite(savename, frame)
        print("Done")


if __name__ == "__main__":

    opt = opts().init()
    opt.load_model = "./CenterTrack/models/coco_pose_tracking.pth"
    opt.save_results = True
    opt.save_video = True

    # video_classifier = "no_clf"
    video_classifier = "manual"
    # video_classifier = "mc3_r2_search03"
    # video_classifier = "r2p_r2_search04"
    # video_classifier = "r3d_r2_search05"
    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")
    detector = CenterTrackDetector(opt, video_classifier)

    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        detector.apply_to_video(video_file)

    # imagefile = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/dump_v2/tackle_bbox/no_clf-detr/vid142_frame00016502/frame00245.jpg"
    # detector = CenterTrackDetector(opt, video_classifier)
    # ret, img = detector.detect(imagefile)
    # basename = os.path.basename(imagefile)
    # cv2.imwrite("./tmp/" + basename, img)