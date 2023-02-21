import os
import pickle
import itertools
from typing import List, Tuple

import cv2
import yaml
import numpy as np
from tqdm import tqdm

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["eval_component"]

limb_pair = [
    (0, 2), (2, 4), 
    (0, 1), (1, 3), 
    (5, 6), (6, 8), (8, 10), 
    (5, 7), (7, 9), 
    (12, 11), (6, 12), (5, 11),
    (12, 14), (14, 16), 
    (11, 13), (13, 15),
]

class TackleClassifier:

    def __init__(self, tackle_classifier: str) -> None:
        """
        Args:

        Returns:
        
        """
        self._load_classifier(tackle_classifier)
        self.classifier_name = tackle_classifier

    def _load_classifier(self, tackle_classifier: str) -> None:
        """
        Args:
            tackle_classifier (str): 
        Returns:
            None
        """
        if tackle_classifier in config["trained_classifier"].keys():
            classifier_file = config["trained_classifier"][tackle_classifier]
            with open(classifier_file, "rb") as fp:
                self.classifier = pickle.load(fp)
        else:
            # dummy classifier for testing.
            from sklearn.dummy import DummyClassifier # temporal
            print("DUMMY CLASSIFIER")
            self.classifier = DummyClassifier(strategy="most_frequent")
            num_sample = 100
            X = np.random.randn(num_sample, 68)
            y = np.random.randint(0, 1, num_sample)
            self.classifier.fit(X, y)

    def _prep_feature(self, pose_pair: Tuple) -> np.ndarray: 
        """
        Args:
            pose_pair (Tuple): Tuple of dictionaries, each containing pose information.
                eg. ({"keypoints": array, "score": float, "area": float}, {})
        Returns:
            feature (np.ndarray): Array of shape 68, (17 <joints> x 2 <x and y axis> x 2 <players>).
                Features are normalized by 
                    1. calculating distance from center points for each coordinates.
                    2. normalize distance by max euclidean distance from center.
                [17 `x` for player1, 17 `y` for player1, 17 `x` for player2, 17 `y` for player2].
        """
        pose_1 = pose_pair[0]["keypoints"][:, :2]
        pose_2 = pose_pair[1]["keypoints"][:, :2]

        # Normalize
        center_x = (pose_1[:, 0].mean() + pose_2[:, 0].mean()) / 2
        center_y = (pose_1[:, 1].mean() + pose_2[:, 1].mean()) / 2
        center = np.array([center_x, center_y])

        # Calc distance from center for each point.
        dist_1 = np.linalg.norm(pose_1 - center, axis=1)
        dist_2 = np.linalg.norm(pose_2 - center, axis=1)
        max_dist = max(dist_1.max(), dist_2.max())

        # Form feature and normalize.
        feature = np.concatenate([
            pose_1[:, 0] - center_x, pose_1[:, 1] - center_y,
            pose_2[:, 0] - center_x, pose_2[:, 1] - center_y,
        ])
        feature = feature / max_dist
        return feature

    def _classify(self, features: List) -> np.ndarray: 
        """
        Args:
            features (List): List of array of shape 68, (17 <joints> x 2 <x and y axis> x 2 <players>).
        Returns:
            result (np.ndarray): 
        """
        result = self.classifier.predict(features)
        return result

    def _draw(self, img, dots):

        for dot in dots:
            cv2.circle(
                img, 
                center=dot.astype(int), 
                radius=5, 
                color=(20,20,20),
                thickness=-1
            )
        for limb_src, limb_dst in limb_pair:
            cv2.line(
                img,
                dots[limb_src].astype(int),
                dots[limb_dst].astype(int),
                color=(255, 255, 255),
                thickness=2,
            )
        return img

    def draw(self, img_idx: int, basename: str, keypoints: List, savename: str):
        tmp_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/dump_v2/tackle_video/no_clf"
        imgfile = os.path.join(tmp_loc, basename, img_idx, "04.jpg")
        img = cv2.imread(imgfile)
        for kp in keypoints:
            img = self._draw(img, kp["keypoints"])
        cv2.imwrite(savename, img)

    def classify_tackle(
        self, 
        video_file: str,
        video_classifier: str,
        tackle_detector: str,
        pose_detector: str,
        ball_detector: str
    ) -> None:
        """
        Args:
            videofile (str): 
            tackle_detector (str): 
            pose_detector (str): 
        Returns:
            None
        """
        basename, _ = os.path.splitext(os.path.basename(video_file))
        setting = f"{video_classifier}-{tackle_detector}-{pose_detector}"
        save_setting = setting + f"-{self.classifier_name}"
        pose_loc = os.path.join(
            config["data_root"], 
            "dump_v2/tackle_pose", 
            setting, 
            basename
        )
        save_loc = os.path.join(
            config["data_root"], 
            config["eval_pose"],
            save_setting, 
            basename
        )

        os.makedirs(save_loc, exist_ok=True)

        # Load pose
        with open(pose_loc + "/tackle_poses.pkl", "rb") as fp:
            poses = pickle.load(fp)

        record = {}
        for img_idx, pose_list in tqdm(poses.items()):
            if pose_list is None:
                continue
            if len(pose_list) < 2:
                continue
            
            features = []
            for pose_pair in itertools.combinations(pose_list, 2):
                features.append(self._prep_feature(pose_pair))
            result = self._classify(features)

            for i, pair in enumerate(list(itertools.combinations(pose_list, 2))):
                if result[i] == 0:
                    continue
                savename = save_loc + f"/{img_idx}/{i:03d}.jpg"
                os.makedirs(save_loc + f"/{img_idx}", exist_ok=True)
                self.draw(img_idx, basename, pair, savename)

            record[img_idx] = result.any()

        # Save result
        # with open(save_loc + "/hia_record.pkl", "wb") as fp:
        #     pickle.dump(record, fp)
                
if __name__ == "__main__":
    from argparse import ArgumentParser
    from glob import glob

    parser = ArgumentParser()
    parser.add_argument('--clf', type=str, default="manual")
    parser.add_argument('--classifier', type=str, default="naive_bayes")
    parser.add_argument('--pose-detector', type=str, default="centertrack")
    parser.add_argument('--tackle-detector', type=str, default="retina",
        help='name of detector model (defined at `resouce/model_info.yaml`')
    parser.add_argument('--ball-detector', type=str, default="")
    parser.add_argument('--device', type=str, default="cuda:3")
    args = parser.parse_args()

    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"
    # video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ATIRA_data/raw"
    video_files = glob(video_loc + "/*.mp4")
    clf = TackleClassifier(args.classifier)
    for i, video_file in enumerate(video_files):
        print(f"Working on {i+1} / {len(video_files)} ...")
        clf.classify_tackle(
            video_file, args.clf, args.tackle_detector, 
            args.pose_detector, args.ball_detector)
    