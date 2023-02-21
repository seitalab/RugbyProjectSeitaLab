import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--clf', type=str, default="manual")
parser.add_argument('--pose-detector', type=str, default="hrnet")
parser.add_argument('--tackle-detector', type=str, default="retina",
    help='name of detector model (defined at `resouce/model_info.yaml`')
parser.add_argument('--ball-detector', type=str, default="")
parser.add_argument('--tackle-classifier', type=str, default="naive_bayes",
    help='name of detector model (defined at `../config.yaml`')
args = parser.parse_args()

command1 = (
    "python 03_extract_tackle_pose.py "
    f"--clf {args.clf} --tackle-detector {args.tackle_detector} "
    f"--pose-detector {args.pose_detector}"
)
command2 = (
    "python 04_apply_classifier_to_pose.py "
    f"--clf {args.clf} --tackle-detector {args.tackle_detector} "
    f"--pose-detector {args.pose_detector} "
)
command3 = (
    "python 05_summarize_detection_result.py "
    f"--clf {args.clf} --tackle-detector {args.tackle_detector} "
    f"--pose-detector {args.pose_detector} "
)
os.system(command1)
os.system(command2)
os.system(command3)