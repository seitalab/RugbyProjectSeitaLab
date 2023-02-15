import os
from datetime import timedelta

import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(1)

cfg_file = "../config.yaml"
with open(cfg_file, "r") as fp:
    config = yaml.safe_load(fp)["video_prep"]

def get_total_frame(video_path: str) -> int:
    """
    Args:
        video_path (str): 
    Returns:
        total_frames (int):
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def make_video_clip(video_path: str, video_id: int, frame_id: int) -> str:
    """
    Args:
        video_path (str): 
        video_id (int): 
        frame_id (int): 
    Returns:
        savename (str): Save name.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    event_sec = frame_id / fps
    start_sec = int(event_sec - (60 * config["clip_length"] / 2)) # event time at center.
    start = str(timedelta(0, int(start_sec)))

    duration = f'00:{config["clip_length"]:02d}:00'

    filename = f"vid{video_id:02d}_frame{frame_id:08d}.mp4"
    savename = os.path.join(config["save_root"], filename)

    command = f"ffmpeg -y -nostats -loglevel 0 -i {video_path} -ss {start} -t {duration} -async 1 -c copy {savename}"
    os.system(command)
    return savename

def prepare_video_clips():
    """
    Args:
        None
    Returns:
        None
    """
    # prepare save loc and summary csv.
    os.makedirs(config["save_root"], exist_ok=True)
    fileinfo = []

    # Load video dict.
    df = pd.read_csv(config["video_id_csv"])
    video_id_dict = dict(zip(df.video_id, df.path))

    # Fixed location.
    print("Working on fixed samples ...")
    event_list = pd.read_csv(config["tackle_event_csv"], index_col=0)
    for _, event in tqdm(event_list.iterrows(), total=len(event_list)):
        video_path = video_id_dict[event["video_id"]]
        video_loc = os.path.join(config["video_root"], video_path)
        savename = make_video_clip(video_loc, event["video_id"], event["frame"])

        label = "hia" if event["is_hia_tackle"] else "non-hia"
        videoname = os.path.basename(savename)
        fileinfo.append([videoname, label])

    # Random.
    print("Working on random samples ...")
    event_list = pd.read_csv(config["tackle_event_csv"], index_col=0)
    for _, event in tqdm(event_list.iterrows(), total=len(event_list)):
        video_path = video_id_dict[event["video_id"]]
        video_loc = os.path.join(config["video_root"], video_path)

        total_frames = get_total_frame(video_loc)
        random_frame_idxs = [
            int(r_idx * total_frames) 
            for r_idx in np.random.rand(config["num_random_per_tackle"])
        ]
        for frame_idx in random_frame_idxs:
            savename = make_video_clip(video_loc, event["video_id"], frame_idx)
            label = "random"
            videoname = os.path.basename(savename)
            fileinfo.append([videoname, label])

    # Save fileinfo csv.    
    df_fileinfo = pd.DataFrame(fileinfo, columns=["videofile", "label"])
    savename = os.path.join(config["save_root"], "videolabels.csv")
    df_fileinfo.to_csv(savename)
    print("Done")

if __name__ == "__main__":

    prepare_video_clips()