import os
import sys
from typing import Dict, List, Tuple

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../model_prep")
from hia_detector import utils
from video_classfier.codes.architectures.model import prepare_model

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["eval_component"]

class TackleVideoClassfier:

    def __init__(
        self, 
        modelname: str, 
        backbone: str, 
        ckptfile_loc: str, 
        device: str
    ) -> None:
        """
        Args:

        Returns:
            None
        """
        self.device = device
        self.modelname = modelname

        if self.modelname not in ["", "no_clf", "manual"]:
            self.model = prepare_model(backbone)
            weight_file = os.path.join(ckptfile_loc, "net.pth")
            self.set_weight(weight_file)

    def set_weight(self, weight_file: str):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def _manual_clf(
        self, 
        video_file: str, 
    ) -> np.ndarray:
        """
        Args:
            video_file (str): 
            idx_blocks (List): 
        Returns:
            is_tackle (np.ndarray): 
        """
        csvfile = os.path.join(
            config["manual_label_csv"], 
            os.path.basename(video_file)[:-4]+".csv"
        )
        df = pd.read_csv(csvfile)
        is_tackle = df.loc[:, "is_tackle"] == 1

        return is_tackle.values.astype(int)

    def _apply(self, video_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            video_file (str): 
        Returns:
            frame_idxs (np.ndarray): 
            tackle_confs (np.ndarray): confidence of tackles
        """
        # Extract frames from video.
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames, frame_idxs = [], []
        for frame_idx in tqdm(range(total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                frame_idxs.append(frame_idx)
        cap.release()

        # Convert frames into chunks.
        frames = np.array(frames)
        frame_idxs = np.array(frame_idxs)
        num_split = total_frames // 5
        frame_blocks = np.split(frames[:num_split*5], num_split)
        idx_blocks = np.split(frame_idxs[:num_split*5], num_split) # array of [(0,1,2,3,4), (...), ...]

        # if no video classifier selected, return all frame data.
        frame_idxs = np.array([i[0] for i in idx_blocks])
        if self.modelname in ["", "no_clf"]:
            tackle_confs = np.ones(len(idx_blocks))
            return frame_idxs, tackle_confs
        elif self.modelname == "manual":
            tackle_confs = self._manual_clf(video_file)
            return frame_idxs, tackle_confs

        # Apply classifier model.
        bs = 16
        is_tackle_video = []
        for i in tqdm(range(0, len(frame_blocks), bs)):
            batch = np.array(frame_blocks[i: i + bs])
            # batch = batch.astype(float) / 255.
            data_tensor = torch.from_numpy(batch)
            # [bs, num_frame, h, w, num_channel] -> [bs, num_channel, num_frame, h, w]
            data_tensor = torch.transpose(data_tensor, 4, 3)
            data_tensor = torch.transpose(data_tensor, 3, 2)
            data_tensor = torch.transpose(data_tensor, 2, 1)
            data_tensor = data_tensor.float().to(self.device)
            # Apply classifier model.
            preds = self.model(data_tensor)
            preds = torch.sigmoid(preds).cpu().detach().numpy()
            is_tackle_video += list(preds)
        
        return frame_idxs, np.array(is_tackle_video)

    def classify_video(self, video_file: str):
        """
        Args:
            video_file (str): 
        Returns:
            None
        """
        # Prep save loc.
        basename, _ = os.path.splitext(os.path.basename(video_file))
        save_loc = os.path.join(
            config["data_root"], 
            config["eval_clf"], 
            self.modelname
        )
        os.makedirs(save_loc, exist_ok=True)

        # Apply model
        frame_idxs, tackle_confs = self._apply(video_file)
        cat = np.stack([frame_idxs, tackle_confs])
        
        # Save frames.
        df = pd.DataFrame(cat.T, columns=["frame_idx", "tackle_confidence"])
        df.loc[:, "frame_idx"] = df.loc[:, "frame_idx"].astype("int")
        savename = save_loc + f"/{basename}.csv"
        df.to_csv(savename)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from glob import glob

    param_file = "../hia_detector/resource/model_info.yaml"

    parser = ArgumentParser()
    parser.add_argument('--clf', type=str, default="mc3_r2_search03", 
        help='name of classifier model (defined at `resouce/model_info.yaml`')
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    with open(param_file) as f:
        params = yaml.safe_load(f)

    if args.clf not in ["", "no_clf", "manual"]:
        backbone = params["video_classifier"][args.clf]["model"]
        ckpt_file = params["video_classifier"][args.clf]["checkpoint_file"]
    else:
        backbone = ""
        ckpt_file = ""
    detector = TackleVideoClassfier(args.clf, backbone, ckpt_file, args.device)

    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"

    video_files = glob(video_loc + "/*.mp4")
    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        detector.classify_video(video_file)