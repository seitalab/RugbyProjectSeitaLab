import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

COLOR_BBOX = (0, 255, 255)
COLOR_KPT = (255, 0, 0)

def get_best_conf_bbox(bboxes: List) -> np.ndarray: 
    """
    Args:
        bboxes (List): 
    Returns:
        best_bbox (np.ndarray): 
    """
    best_conf_bbox_idx = np.argmax(bboxes[0][:, 4])
    best_bbox = bboxes[0][best_conf_bbox_idx]
    return best_bbox

def draw_top_bbox(img: np.ndarray, bboxes: List) -> np.ndarray:
    """
    Args:
        img (np.ndarray): 
        bboxes (List): List of np.ndarray.
    Returns:
        img (np.ndarray): 
    """
    best_bbox = get_best_conf_bbox(bboxes).astype(int)

    bbox_lt = (best_bbox[0], best_bbox[1])
    bbox_rb = (best_bbox[2], best_bbox[3])
    cv2.rectangle(img, bbox_lt, bbox_rb, color=COLOR_BBOX, thickness=7)
    return img

def draw_keypoints(img: np.ndarray, keypoints: Dict) -> np.ndarray:
    """
    Args:
        img (np.ndarray): 
        bboxes (List): List of np.ndarray.
    Returns:
        img (np.ndarray): 
    """
    for point in keypoints["keypoints"]:
        center = point[:2].astype(int)
        cv2.circle(img, center, radius=3, color=COLOR_KPT, thickness=-1)
    return img

def eval_score(gt_labels: np.ndarray, hia_prediction: np.ndarray) -> Tuple[float, bool]: 
    """
    Args:

    Returns:
        score
    """
    # temporal
    tp_score = 300
    fn_score = -300
    tn_score = 0
    fp_score = -0.1

    score = 0
    window = gt_labels == 1
    pred_in_window = hia_prediction[window]
    pred_out_window = hia_prediction[~window]

    # Add score in window.
    if pred_in_window.size != 0:
        hia_predicted = (pred_in_window == 1).any()
        score += hia_predicted * tp_score + ~hia_predicted * fn_score
    else:
        hia_predicted = hia_prediction.any()

    # Add score of outside window.
    score += pred_out_window.sum() * fp_score
    score += (1 - pred_out_window).sum() * tn_score
    
    # Normalize.
    score /= len(gt_labels)
    return score, hia_predicted

def edit_window(gt_labels: np.ndarray, hia_prediction: np.ndarray):
    """
    Args:

    Returns:

    """
    window = gt_labels == 1
    pred_in_window = hia_prediction[window]

    pred_edit = np.zeros_like(hia_prediction)
    pred_edit[~window] = hia_prediction[~window]

    if pred_in_window.size != 0:
        hia_predicted = (pred_in_window == 1).any()
        pred_edit[window] = int(hia_predicted)
    
    return pred_edit

def eval_score_tot(gt_labels: np.ndarray, hia_prediction: np.ndarray):
    """
    Args:

    Returns:

    """
    tp_score = 1
    fn_score = -1
    fp_score = -0.1
    tn_score = 0

    tp = ((gt_labels == 1) & (hia_prediction == 1)).astype(int)
    fn = ((gt_labels == 1) & (hia_prediction == 0)).astype(int)
    fp = ((gt_labels == 0) & (hia_prediction == 1)).astype(int)
    tn = ((gt_labels == 0) & (hia_prediction == 0)).astype(int)

    u_opt = gt_labels.sum()
    u_none = fn.sum() * fn_score + fp.sum() * fp_score
    u_total = fn.sum() * fn_score + tp.sum() * tp_score + fp.sum() * fp_score
    u_score = (u_total - u_none) / (u_opt - u_none)
    return u_score

def prepare_dirpath(
    video_file: str,
    video_classifier: str,
    tackle_detector: str,
    pose_detector: str,
    ball_detector: str,
    classifier_name: str,
    mode: str,
    config: Dict,
):
    """
    Args:

    Returns:

    """
    basename, _ = os.path.splitext(os.path.basename(video_file))
    if video_classifier == "":
        video_classifier = "no_clf"

    if mode == "apply_video_classifier":
        
        save_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            video_classifier, basename
        )

        return save_loc

    elif mode == "apply_tackle_detector":
        assert tackle_detector is not None
        assert pose_detector is None
        setting = f"{video_classifier}-{tackle_detector}"

        save_loc = os.path.join(
            config["data_root"], 
            config["tackle_frame"]["save_loc"], 
            setting, 
            basename
        )

        load_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            video_classifier, 
            basename
        )
        return save_loc, load_loc

    elif mode == "apply_pose_detector":
        assert tackle_detector is None
        assert pose_detector is not None
        setting = f"{video_classifier}-{pose_detector}"
        
        save_loc = os.path.join(
            config["data_root"], 
            config["pose_detect"]["save_loc"], 
            setting, 
            basename
        )

        load_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            video_classifier, 
            basename
        )
        return save_loc, load_loc

    elif mode == "apply_ball_detector":
        assert ball_detector is not None
        assert ball_detector != ""
        setting = f"{video_classifier}-{ball_detector}"

        save_loc = os.path.join(
            config["data_root"], 
            config["ball_frame"]["save_loc"], 
            setting, 
            basename
        )

        load_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"], 
            video_classifier, 
            basename
        )
        return save_loc, load_loc

    elif mode == "extract_pose":
        setting = f"{video_classifier}-{tackle_detector}-{pose_detector}"

        kpt_loc = os.path.join(
            config["data_root"], 
            config["pose_detect"]["save_loc"], 
            f"{video_classifier}-{pose_detector}", 
            basename
        )
        bbox_loc = os.path.join(
            config["data_root"], 
            config["tackle_frame"]["save_loc"], 
            f"{video_classifier}-{tackle_detector}", 
            basename
        )
        if ball_detector != "":
            ball_box_loc = os.path.join(
                config["data_root"], 
                config["ball_detect"]["save_loc"], 
                f"{video_classifier}-{ball_detector}", 
                basename
            )
            setting += f"-{ball_detector}"
        else:
            ball_box_loc = None

        save_loc = os.path.join(
            config["data_root"], 
            config["tackle_pose_extract"]["save_loc"], 
            setting, 
            basename
        )
        return kpt_loc, bbox_loc, ball_box_loc, save_loc

    elif mode == "apply_tackle_classifier":
        setting = f"{video_classifier}-{tackle_detector}-{pose_detector}"
        if ball_detector != "":
            setting += f"-{ball_detector}"
        save_setting = setting + f"-{classifier_name}"
        pose_loc = os.path.join(
            config["data_root"], 
            config["tackle_pose_extract"]["save_loc"], 
            setting, 
            basename
        )
        save_loc = os.path.join(
            config["data_root"], 
            config["tackle_classifier"]["save_loc"], 
            save_setting, 
            basename
        )
        return pose_loc, save_loc

