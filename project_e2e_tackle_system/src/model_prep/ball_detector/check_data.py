import cv2
from os import path as osp

dump_loc = "./dump"
# target_file = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ball/processed/base/train/0057.jpg"
# target_file = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ball/processed/base/train/0235.jpg"
# target_file = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ball/processed/base/valid/0147.jpg"
target_file = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ball/processed/base/test/0099.jpg"
basename = osp.basename(target_file)
bbox_file = target_file.replace(".jpg", ".txt")


img = cv2.imread(target_file)
h, w, _ = img.shape

with open(bbox_file) as f:
    bboxes = f.read().strip()

bboxes = bboxes.split("\n")
for bbox in bboxes:
    bbox = bbox.split(" ")
    bbox = [float(e) for e in bbox[1:]]

    bbox = [
            bbox[0] - bbox[2]/2, 
            bbox[1] - bbox[3]/2, 
            bbox[0] + bbox[2]/2, 
            bbox[1] + bbox[3]/2
        ]
    bbox = [int(bbox[0] * w), int(bbox[1] * h),
            int(bbox[2] * w), int(bbox[3] * h)]
    
    cv2.rectangle(
        img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        color=(0, 255, 0), thickness=5
    )
cv2.imwrite(osp.join(dump_loc, basename), img)
