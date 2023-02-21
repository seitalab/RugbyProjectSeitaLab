import os
from glob import glob

import cv2
import yaml
import mmcv
from tqdm import tqdm

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["ball_detector"]

def convert_to_coco(datatype: str, mode: str) -> None:
    """
    Args:
        datatype (str):
    Returns:
        None
    """
    print(f"Working on {datatype} ...")
    data_loc = os.path.join(config["processed_data_loc"], mode, datatype)
    imgfiles = sorted(glob(data_loc + "/*.jpg"))

    anno_idx = 0

    data_dict = {"images": [], "annotations": [], "categories": []}
    data_dict["categories"].append({"id": 0, "name": "ball"})

    for idx, imgfile in enumerate(tqdm(imgfiles)):
        img_idx = int(os.path.basename(imgfile)[:-4])

        img = cv2.imread(imgfile)
        h, w, _ = img.shape
        basename = os.path.basename(imgfile)
        img_dict = {
            "id": idx, 
            "file_name": basename, 
            "height": h, 
            "width": w
        }
        data_dict["images"].append(img_dict)

        bboxfile = imgfile.replace(".jpg", ".txt")
        bbox = open(bboxfile, "r").read()
        
        # Skip if random image with no bbox.
        if bbox == " ": 
            continue
        
        if img_idx in config["errors"]["fix"]:
            bbox = bbox.split("\n")[0]
        if img_idx in config["errors"]["skip"]:
            continue

        bbox = bbox.split(" ")
        assert len(bbox) == 5 # Only 1 bbox per file.

        class_label = int(bbox[0])
        bbox = [float(e) for e in bbox[1:]]
        # (center_x, center_y, w, h) -> (top_left_x, top_left_y, w, h).
        bbox = [
            bbox[0] - bbox[2]/2, 
            bbox[1] - bbox[3]/2, 
            bbox[2], 
            bbox[3]
        ]
        bbox = [int(bbox[0] * w), int(bbox[1] * h),
                int(bbox[2] * w), int(bbox[3] * h)]
        area = (bbox[0] + bbox[2]) * (bbox[1] + bbox[3])

        anno_dict = {
            "image_id": idx, "id": anno_idx, "category_id": class_label,
            "bbox": bbox, "area": area, "iscrowd": 0}
        anno_idx += 1
        data_dict["annotations"].append(anno_dict)

    savename = data_loc + "/annotation_coco.json"
    mmcv.dump(data_dict, savename)

if __name__ == "__main__":

    mode = "base"
    convert_to_coco("train", mode)
    convert_to_coco("valid", mode)
    convert_to_coco("test", mode)