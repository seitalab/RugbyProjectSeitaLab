import json
from datetime import datetime
from typing import Any, List, Dict, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
import torch
import torchvision.ops.boxes as bops
from sklearn.metrics import classification_report

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["tackle_detector"]["detector_train"]["utils"]

JsonDict = Dict[str, Any]

def extract_target_bbox(json_data: JsonDict, target_idx: int) -> List:
    """
    Extract bbox of target_idx from json data.
    Args:
        json_data:
        target_idx (int):
    Returns:
        bbox (np.array): array of length 4.
    """
    bbox = None
    for json_item in json_data:
        if json_item['tracking_id'] == target_idx:
            bbox = json_item['bbox']
    # assert bbox is not None
    return bbox

def prepare_bbox(json_path: str, carrier_idx: int, tackler_idx: int) -> List:
    """
    Args:
        json_path (str):
        carrier_idx (int):
        tackler_idx (int):
    Returns:
        bbox (List): Aggregated bbox.
    """
    # Load json.
    json_data = json.load(open(json_path, "r"))

    # Extract bbox.
    bbox_carrier = extract_target_bbox(json_data, carrier_idx)
    bbox_tackler = extract_target_bbox(json_data, tackler_idx)
    # if (bbox_carrier is None or bbox_tackler is None):
    #     print(bbox_tackler)
    #     print(bbox_carrier)
    #     return None
    # else:
    #     return aggregate_bbox(bbox_carrier, bbox_tackler)
    if (bbox_carrier is None and bbox_tackler is None):
        return None
    elif bbox_carrier is None:
        return bbox_tackler
    elif bbox_tackler is None:
        return bbox_carrier
    else:
        # Process bbox.
        return aggregate_bbox(bbox_carrier, bbox_tackler)

def aggregate_bbox(bbox_carrier: List, bbox_tackler: List) -> List:
    """
    Aggregate bbox of carrier and tackler.

    Args:
        bbox_carrier (List):
        bbox_tackler (List):
    Returns:
        bbox (List):
    """
    left_x = min([bbox_carrier[0], bbox_tackler[0]])
    right_x = max([bbox_carrier[2], bbox_tackler[2]])
    top_y = min([bbox_carrier[1], bbox_tackler[1]])
    bottom_y = max([bbox_carrier[3], bbox_tackler[3]])

    bbox = [left_x, top_y, right_x, bottom_y]
    return bbox

def scale_bbox(bbox: List, image_path: str) -> List:
    """
    Scale size of bbox by image size.
        x: bbox horizontal center.
        y: bbox vertical center.
        w: bbox width.
        h: bbox height.

    Args:
        bbox (List):
        image_path (str):
    Returns:
        scaled_bbox
    """
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    scaled_bbox = [
        (bbox[0] + bbox[2])/2/img_w, (bbox[1] + bbox[3])/2/img_h,
        (bbox[2] - bbox[0])/img_w, (bbox[3] - bbox[1])/img_h]
    return scaled_bbox

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def extract_label(bboxfile: str) -> int:
    """
    Args:
        bboxfile (str):
    Returns
        class_idx (int):
    """
    bbox = open(bboxfile, "r").read()
    if bbox.strip() == "":
        return None, None
    return int(bbox.split(" ")[0])

def load_bbox(bboxfile: str, image_w: int, image_h: int):
    """
    Args:
        bboxfile (str):
        image_w (int):
        image_h (int):
    Returns:

    """
    bbox = open(bboxfile, "r").read()
    if bbox.strip() == "":
        return None, None
    bbox = [float(v) for v in bbox.split(" ")[1:]]
    bbox = np.array(bbox) * np.array([image_w, image_h, image_w, image_h])
    x1 = int(bbox[0] - (bbox[2] / 2))
    y1 = int(bbox[1] - (bbox[3] / 2))
    x2 = int(x1 + bbox[2])
    y2 = int(y1 + bbox[3])
    return (x1, y1), (x2, y2)

def draw_bbox(
    img_data: np.ndarray,
    bbox_loc: Tuple,
    class_idx: int=0,
    is_gt: bool=False
) -> np.ndarray:
    """
    Args:
        img_data (np.ndarray):
        bbox_loc (Tuple): Tuple of (left_top(x, y), right_bottom(x,y)).
        class_idx (int):
        is_gt (bool):
    Returns:
        img_data
    """
    left_top, right_bottom = bbox_loc
    if is_gt:
        if class_idx == 0:
            color = config["GT_COLOR"]
        # elif class_idx == 1:
        #     color = GT_COLOR_class1
        else:
            raise NotImplementedError
    else:
        if class_idx == 0:
            color = config["BBOX_COLOR"]
        # elif class_idx == 1:
        #     color = BBOX_COLOR_class1
        else:
            raise NotImplementedError
    img_data = cv2.rectangle(
        img_data, 
        left_top, 
        right_bottom, 
        color=color, 
        thickness=config["thickness"]
    )
    return img_data

def add_text_box(bbox: Tuple, text: str, img_data: np.ndarray) -> np.ndarray:
    """
    Args:
        bbox (Tuple):
        text (str):
        img_data (np.ndarray):
    Returns:
        img_data (np.ndarray):
    """
    x, y = bbox[0] # lefttop of bbox
    w = max(bbox[1][0] - x, len(text))
    cv2.rectangle(
        img_data, 
        (x, y-20), 
        (x + w, y), 
        color=config["TEXT_BACK_COLOR"], 
        thickness=-1
    )
    cv2.putText(
        img_data, 
        text, 
        (x, y-10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.3, 
        config["TEXT_COLOR"]
    )
    return img_data

def calc_score(gt_bbox, pred_bboxes) -> np.ndarray:
    """
    Args:
        gt_bbox ():
        pred_bboxes ()
    Returns:
        scores (np.ndarray):
    """
    if gt_bbox[0] is None:
        return []
    if pred_bboxes == []:
        return []

    box1 = torch.tensor([gt_bbox[0] + gt_bbox[1]], dtype=torch.float)
    scores = []
    for pred_bbox in pred_bboxes:
        pred_bbox = np.concatenate([pred_bbox[0], pred_bbox[1]])
        box2 = torch.tensor([pred_bbox], dtype=torch.float)
        iou = bops.box_iou(box1, box2)
        scores.append(float(iou[0][0]))
    scores = np.array(scores)
    return scores

def calc_detector_score(df_result: pd.DataFrame) -> str:
    """
    Args:

    Returns:

    """
    gt = df_result.loc[:, "gt_bbox"].values.astype(int)

    pred = df_result.loc[:, "num_bbox"].values.astype(int)
    pred = (pred > 0).astype(int)

    report = classification_report(gt, pred)
    tpr = (gt * pred).sum() / gt.sum()
    fpr = ((1 - gt) * pred).sum() / (1 - gt).sum()
    report += "\n" + "-"*80 + "\n"
    report += f"True Positive Rate: {tpr:.03f}\n"
    report += f"False Positive Rate: {fpr:.03f}\n"
    return report

    # precision = precision_score(gt, pred)
    # recall = recall_score(gt, pred)
    # f1 = f1_score(gt, pred)
    # conf_matrix = confusion_matrix
