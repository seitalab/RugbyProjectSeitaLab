import os
import pickle
from typing import List, Tuple

import yaml
import numpy as np
from tqdm import tqdm

import utils

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["hia_detector"]

def isin_bbox(bbox: np.ndarray, keypoints: np.ndarray):
    """
    Args:
        bbox (np.ndarray): Array of bbox info eg. (lt_x, lt_y, rb_x, rb_y).
        keypoints (np.ndarray): Array of keypoint of human joints.
    Returns:
        isin (bool): 
    """
    # Use only main body (= body trunk) area.
    mainbody_idx = np.array([5, 6, 11, 12])
    keypoints = keypoints[mainbody_idx]
    
    isin_lt_x = keypoints[:, 0] >= bbox[0]
    isin_lt_y = keypoints[:, 1] >= bbox[1]
    isin_rb_x = keypoints[:, 0] <= bbox[2]
    isin_rb_y = keypoints[:, 1] <= bbox[3]
    isin = isin_lt_x & isin_rb_x & isin_lt_y & isin_rb_y
    return isin.any()

def select_kpt_in_bbox(bboxes: List, keypoints: Tuple) -> List:
    """
    Args:
        bboxes (List): 
        keypoints (Tuple): Tuple of list eg. ([{}, {}, ..., {}])
    Returns:
        selected_keypoints (List): 
    """
    selected_keypoints = []

    if utils.get_best_conf_bbox(bboxes)[4] < config["bbox_threshold"]:
        return None
    best_bbox = utils.get_best_conf_bbox(bboxes)

    for keypoint_group in keypoints:
        if keypoint_group == {}:
            continue
        if isin_bbox(best_bbox, keypoint_group["keypoints"]):
            selected_keypoints.append(keypoint_group)
    return selected_keypoints

def eval_bbox_overlap(tackle_bboxes: List, ball_bboxes: List) -> bool:
    """
    Args:

    Returns:

    """
    pass

def extract_tackle_pose(
    video_file: str,
    video_classifier: str,
    tackle_detector: str, 
    pose_detector: str,
    ball_detector: str
) -> None:
    """
    Args:
        video_file (str): 
        video_classifier (str): 
        tackle_detector (str): Name of tackle detector.
        pose_detector (str): Name of pose detector.
        ball_detector (Optional[str]): Name of ball detector.
    Returns:
        None
    """
    # load setting.
    kpt_loc, bbox_loc, ball_box_loc, save_loc = utils.prepare_dirpath(
            video_file=video_file,
            video_classifier=video_classifier,
            tackle_detector=tackle_detector,
            pose_detector=pose_detector,
            ball_detector=ball_detector,
            classifier_name=None,
            mode="extract_pose",
            config=config,
        )
    os.makedirs(save_loc, exist_ok=True)

    # Load dict.
    with open(bbox_loc + "/bboxes.pkl", "rb") as fpb:
        bboxes = pickle.load(fpb)
    with open(kpt_loc + "/keypoint_dict.pkl", "rb") as fpk:
        kpt_dict = pickle.load(fpk)
    
    if ball_detector != "":
        with open(ball_box_loc + "/bboxes.pkl", "rb") as fpbb:
            ball_bboxes = pickle.load(fpbb)
    
    # Select keypoints.
    selected_point_dict = {}
    for img_idx in tqdm(kpt_dict.keys()):
        if bboxes[img_idx][0].size == 0:
            continue
        if pose_detector == "humancheck":
            selected = kpt_dict[img_idx]
        else:
            selected = select_kpt_in_bbox(bboxes[img_idx], kpt_dict[img_idx])
        if ball_detector != "":
            selected_ball = eval_bbox_overlap(
                bboxes[img_idx], ball_bboxes[img_idx])
        selected_point_dict[img_idx] = selected

    with open(save_loc + "/tackle_poses.pkl", "wb") as fp:
        pickle.dump(selected_point_dict, fp)

if __name__ == "__main__":
    from glob import glob
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--clf', type=str, default="mc3_v1")
    parser.add_argument('--pose-detector', type=str, default="hrnet")
    parser.add_argument('--tackle-detector', type=str, default="retina",
        help='name of detector model (defined at `resouce/model_info.yaml`')
    parser.add_argument('--ball-detector', type=str, default="")
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")
    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        extract_tackle_pose(
            video_file, args.clf, args.tackle_detector,   
            args.pose_detector, args.ball_detector)