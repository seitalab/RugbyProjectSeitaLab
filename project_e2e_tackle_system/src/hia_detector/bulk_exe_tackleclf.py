import os
from itertools import product

clfs = [
    "manual", 
    "mc3_r2_search03", 
    "r2p_r2_search04", 
    "r3d_r2_search05", 
    "no_clf"
]
pose_detectors = [
    "centertrack",
    "hrnet",
    "humancheck"
]
tackle_detectors = [
    "retina",
    "detr"
]

triplets = list(product(*[clfs, pose_detectors, tackle_detectors]))

for i, (clf, pose_detector, tackle_detector) in enumerate(triplets):
    print(f"{i+1}/{len(triplets)} ...")

    command = (
        "python 04_apply_classifier_to_pose.py "
        f"--clf {clf} --tackle-detector {tackle_detector} "
        f"--pose-detector {pose_detector} "
    )
    os.system(command)