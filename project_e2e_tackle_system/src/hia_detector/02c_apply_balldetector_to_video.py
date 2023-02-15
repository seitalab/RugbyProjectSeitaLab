import os
import pickle
from glob import glob
from typing import Dict

import cv2
import yaml
from tqdm import tqdm

import utils
from apply_detector_to_video import TackleFrameDetector
from mmdet.apis import init_detector

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["hia_detector"]

class BallFrameDetector(TackleFrameDetector):

    def __init__(
        self, 
        video_classifier: str,
        modelname: str, 
        config_file: str, 
        checkpoint_file: str, 
        device: str
    ) -> None:
        """
        Args:
            modelname (str): 
            config_file (str): 
            checkpoint_file (str): 
            device (str): 
        Returns:
            None
        """        
        self.video_classifier = video_classifier
        self.model = init_detector(
            config_file, checkpoint_file, device=device)
        self.ball_detector = modelname

    def detect_ball(self, video_file: str):
        """
        Args:
            video_file (str): 
        Returns:
            None
        """
        # Prep path.
        save_loc, load_loc = utils.prepare_dirpath(
            video_file=video_file,
            video_classifier=self.video_classifier,
            tackle_detector=None,
            pose_detector=None,
            ball_detector=self.ball_detector,
            classifier_name=None,
            mode="apply_ball_detector",
            config=config,
        )
        os.makedirs(save_loc, exist_ok=True)
        frame_block_locs = glob(load_loc + "/frame*")

        # Apply model
        bboxes, images = self._apply(frame_block_locs)

        # Save bboxes.
        with open(save_loc + "/bboxes.pkl", "wb") as fpb:
            pickle.dump(bboxes, fpb)

        # Save frames.
        print("Saving frames ...")
        for frame_idx in tqdm(images.keys()):
            bbox = bboxes[frame_idx]
            frame = images[frame_idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            savename_plain = save_loc + f"/{frame_idx}.jpg"
            cv2.imwrite(savename_plain, frame)

            if bbox == []:
                continue
            # Draw bbox if above threshold.
            if bbox[0].size == 0:
                continue
            if utils.get_best_conf_bbox(bbox)[4] > config["bbox_threshold"]:
                frame = utils.draw_top_bbox(frame, bbox)

                savename_bbox = save_loc + f"/{frame_idx}_bbox.jpg"
                cv2.imwrite(savename_bbox, frame)

        print("Done")

if __name__ == "__main__":
    from argparse import ArgumentParser
    from glob import glob

    param_file = "./resource/model_info.yaml"

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="retina_ball_v1", 
        help='name of detector model (defined at `resouce/model_info.yaml`')
    parser.add_argument('--clf', type=str, default="mc3_v1")
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    with open(param_file) as f:
        params = yaml.safe_load(f)

    config_file = params["ball_detect"][args.model]["config_file"]
    ckpt_file = params["ball_detect"][args.model]["checkpoint_file"]
    detector = BallFrameDetector(
        args.clf, args.model, config_file, ckpt_file, args.device)

    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")
    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        detector.detect_ball(video_file)
