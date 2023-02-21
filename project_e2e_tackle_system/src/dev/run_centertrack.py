import os
import sys
import pickle
from glob import glob
from typing import List, Dict

import cv2
import yaml
import numpy as np
import pandas as pd
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

    def __init__(self, opt, video_clf: str, use_manual: bool=False):

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.debug = max(opt.debug, 1)
        self.thres = config["pose_threshold"]

        self.video_classifier = video_clf
        self.use_manual = use_manual
        if use_manual:
            self.pose_detector = "humancheck"
            self.detector = None
        else:
            self.pose_detector = "centertrack"
            self.detector = Detector(opt)

    def _draw(self, img: np.ndarray, det: Dict, key_name: str="hps"):
        """
        Draw skeleton on img.

        Args:
            img (np.ndarray): 
            det (Dict): 
        Returns:
            img (np.ndarray)
        """
        # (34, ) -> (17, 2)
        dots = det[key_name]
        if key_name == "hps":
            dots = dots.reshape(-1, 2)
        for dot in dots:
            cv2.circle(
                img, 
                center=dot.astype(int), 
                radius=5, 
                color=(20,20,20),
                thickness=-1
            )
        for limb_src, limb_dst in limb_pair:
            cv2.line(
                img,
                dots[limb_src].astype(int),
                dots[limb_dst].astype(int),
                color=(255, 255, 255),
                thickness=2,
            )
        return img

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

    def _manual_select(self, basename: str, frame_idx: str) -> List: 
        """
        Args:
            basename (str): 
            frame_idx (str): `frame{idx:05d}`
        Returns:
            selected_keypoints (List[Dict]): 
        """
        csvfile = os.path.join(
            config["manual_poseidx_csv"], 
            basename + ".csv"
        )
        df = pd.read_csv(csvfile)
        frame_idx_num = int(frame_idx[5:])
        target_row = df[df.loc[:, "frame_num"] == frame_idx_num]
        # print(basename, frame_idx)
        assert len(target_row) == 1 # no overlap of frame_idx_num
        
        selected_keypoints = []
        tackler_id = int(target_row["tackler_id"].values[0])
        if tackler_id != -1:
            selected_keypoints.append(self.kp_dict[frame_idx][tackler_id])
        carrier_id = int(target_row["carrier_id"].values[0])
        if carrier_id != -1:
            selected_keypoints.append(self.kp_dict[frame_idx][carrier_id])

        return selected_keypoints

    def prep_pose(self, basename: str) -> None:
        """
        Args:
            basename (str): 
        Returns:
            
        """
        kp_dict_file = os.path.join(
            config["manual_poselabel_src"], 
            basename,
            "keypoint_dict.pkl"
        )
        with open(kp_dict_file, "rb") as fp:
            kp_dict = pickle.load(fp)
        
        self.kp_dict = kp_dict

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
            config["data_root"],
            config["pose_detect"]["save_loc"], 
            f"{self.video_classifier}-{self.pose_detector}",
            basename
        )
        load_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            self.video_classifier, 
            basename
        )
        os.makedirs(save_loc, exist_ok=True)

        if self.use_manual:
            self.prep_pose(basename)

        keypoint_dict, images = {}, {}
        frame_block_locs = glob(load_loc + "/frame*")
        for frame_block_loc in tqdm(frame_block_locs):
            # extract frame id.
            frame_id = os.path.basename(frame_block_loc)

            # fetch last frame file.
            frame_file = sorted(glob(frame_block_loc + "/*.jpg"))[-1]
            if self.use_manual:
                # frame_idx = int(frame_file.split("/")[-2][5:]) # .../frame{idx:05d}/xxx.jpg -> idx
                # keypoints, img = self.detect(frame_file, False)
                # keypoints = self._manual_select(basename, frame_idx, keypoints)
                frame_idx = frame_file.split("/")[-2] # .../frame{idx:05d}/xxx.jpg -> idx
                keypoints = self._manual_select(basename, frame_idx)
                img = cv2.imread(frame_file)
                for kp in keypoints:
                    self._draw(img, kp, key_name="keypoints")
            else:
                keypoints, img = self.detect(frame_file, True)
                keypoints = list(
                    filter(lambda x: x["score"] > config["pose_threshold"], keypoints))

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

    use_manual = False

    # video_classifier = "no_clf"
    video_classifier = "manual"
    # video_classifier = "mc3_r2_search03"
    # video_classifier = "r2p_r2_search04"
    # video_classifier = "r3d_r2_search05"
    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")
    detector = CenterTrackDetector(opt, video_classifier, use_manual)

    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        detector.apply_to_video(video_file)

    # imagefile = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/dump_v2/tackle_bbox/no_clf-detr/vid142_frame00016502/frame00245.jpg"
    # detector = CenterTrackDetector(opt, video_classifier)
    # ret, img = detector.detect(imagefile)
    # basename = os.path.basename(imagefile)
    # cv2.imwrite("./tmp/" + basename, img)