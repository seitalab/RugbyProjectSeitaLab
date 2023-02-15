import os
import pickle
from glob import glob
from typing import List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set()

import utils

cfg_file = "../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["hia_detector"]
    
class ResultSummarizer:

    def __init__(
        self, 
        video_classifier: str,
        tackle_detector: str, 
        pose_detector: str,
        ball_detector: str,
        tackle_classifier: str,
    ) -> None:
        """
        Args:

        Returns:

        """
        if video_classifier == "":
            video_classifier = "no_clf"
            
        self.seqlen = 300 # eval_every = 5 / fps = 25 / clip_length = 1min
        self.window = 0.05 # temporal: window size ratio of ground truth.
        self.pose_detector = f"{video_classifier}-{pose_detector}"
        self.tackle_detector = f"{video_classifier}-{tackle_detector}"
        self.classifier_name = tackle_classifier
        self.video_classifier = video_classifier

        self.pose_loc = f"{video_classifier}-{tackle_detector}-{pose_detector}"
        if ball_detector != "":
            self.pose_loc += f"-{ball_detector}"
        self.pose_clf_loc = f"{self.pose_loc}-{tackle_classifier}"

        colors = ((0, 0, 0, 0), (0.9, 0.0, 0.0, 0.75))
        self.cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

        labelfile = os.path.join(
            config["data_root"], "video_clips/videolabels.csv")
        self.df_labels = pd.read_csv(labelfile, index_col=0)

        self.save_loc = os.path.join(
            config["data_root"], 
            config["summary"]["save_loc"], 
            self.pose_clf_loc
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def make_figure(self, result: np.ndarray, savename: str, title: str) -> None:
        """
        Args:
            result (np.ndarray): 
            savename (str): 
            title (str): 
        Returns:
            None
        """
        seqlen = result.shape[1]

        plt.figure(figsize=(30, 10))
        ax = sns.heatmap(result, linewidth=0.5, cmap=self.cmap)

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25, 0.75])
        colorbar.set_ticklabels(['Negative', 'Positive'])

        plt.yticks(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
            [
                "GT: HIA", "GT: Tackle", 
                "Classification: Tackle",
                "Detection: Tackle", 
                "Detection: Pose", 
                "Classification: HIA"
            ], 
            rotation=0
        )
        plt.xticks(
            np.arange(seqlen+1)[::50], 
            (np.arange(seqlen+1)[::50]/5).astype(int), 
            rotation=0
        )
        plt.title(title, fontsize=18)
        plt.xlabel("[sec]")
        plt.savefig(savename)
        plt.close()

    def _prepare_ground_truth(self, video_file: str) -> np.ndarray:
        """
        Args:
            video_file (str): 
        Returns:
            gt_labels (np.ndarray): 
        """
        is_target_row = self.df_labels.loc[:, "videofile"] == video_file
        label = self.df_labels.loc[is_target_row, "label"].values[0]

        gt_labels = np.zeros([2, self.seqlen])

        if label != "random":
            # total of 6 seconds as ground truth.
            gt_start = int(self.seqlen / 2 - self.seqlen * self.window / 2)
            gt_end = int(self.seqlen / 2 + self.seqlen * self.window / 2)
            gt_labels[1, gt_start:gt_end] = 1
            if label == "hia":
                gt_labels[0, gt_start:gt_end] = 1
        return gt_labels

    def summarize(self, video_file: str) -> List:
        """
        Args:
            video_file (str): 
        Returns:
            result_data
        """
        basename, _ = os.path.splitext(os.path.basename(video_file))

        tackle_frame_loc = os.path.join(
            config["data_root"], 
            config["target_video"]["save_loc"],
            self.video_classifier,
            basename
        )
        tackle_frames = glob(tackle_frame_loc + "/frame*")

        bbox_loc = os.path.join(
            config["data_root"], 
            config["tackle_frame"]["save_loc"], 
            self.tackle_detector, 
            basename
        )
        with open(bbox_loc + "/bboxes.pkl", "rb") as fp:
            bboxes = pickle.load(fp)

        pose_loc = os.path.join(
            config["data_root"], 
            config["tackle_pose_extract"]["save_loc"], 
            self.pose_loc, basename
        )
        with open(pose_loc + "/tackle_poses.pkl", "rb") as fp:
            poses = pickle.load(fp)
        
        clf_loc = os.path.join(
            config["data_root"], 
            config["tackle_classifier"]["save_loc"], 
            self.pose_clf_loc, basename
        )
        with open(clf_loc + "/hia_record.pkl", "rb") as fp:
            hia_record = pickle.load(fp)

        result = np.zeros([4, self.seqlen])
        for frame_loc in tackle_frames:
            idx = os.path.basename(frame_loc)
            idx_val = int(idx.replace("frame", ""))
            result[0, int(idx_val/5)] = 1

        for idx in bboxes.keys():
            if bboxes[idx][0].size == 0:
                continue
            best_bbox = utils.get_best_conf_bbox(bboxes[idx])
            idx_val = int(idx.replace("frame", ""))
            result[1, int(idx_val/5)] = int(best_bbox[4] > config["bbox_threshold"])
        
        for idx in poses.keys():
            if poses[idx] is None:
                continue
            if len(poses[idx]) == 0:
                continue
            idx_val = int(idx.replace("frame", ""))
            keypoints = list(
                filter(lambda x: x["score"] > config["pose_threshold"], poses[idx]))
            result[2, int(idx_val/5)] = int(len(keypoints) > 0)
        
        for idx in hia_record.keys():
            idx_val = int(idx.replace("frame", ""))
            result[3, int(idx_val/5)] = int(hia_record[idx])

        # Prepare GT labels.
        gt_labels = self._prepare_ground_truth(os.path.basename(video_file))
        result = np.concatenate([gt_labels, result])

        score, hia_predicted = utils.eval_score(result[0], result[-1]) # gt_labels, hia_predictions.
        title = f"{basename} | score: {score:.3f}"

        # print(result.sum(axis=1))
        fig_name = self.save_loc + f"/{basename}.png"
        self.make_figure(result, fig_name, title)

        is_hia = (result[0] > 0).any()
        result_data = [basename, int(is_hia), int(hia_predicted), round(score, 3)]
        pred_edit = utils.edit_window(result[0], result[-1])
        return result_data, (result[0], pred_edit)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from glob import glob

    import yaml
    from tqdm import tqdm

    param_file = "./resource/model_info.yaml"
    video_loc = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/video_clips"

    parser = ArgumentParser()
    parser.add_argument('--clf', type=str, default="manual")
    parser.add_argument('--pose-detector', type=str, default="humancheck")
    parser.add_argument('--tackle-detector', type=str, default="retina",
        help='name of detector model (defined at `resouce/model_info.yaml`')
    parser.add_argument('--ball-detector', type=str, default="")
    parser.add_argument('--tackle-classifier', type=str, default="naive_bayes",
        help='name of detector model (defined at `../config.yaml`')
    args = parser.parse_args()

    with open(param_file) as f:
        params = yaml.safe_load(f)

    video_files = glob(video_loc + "/*.mp4")
    result = []
    gts, preds = [], []
    summarizer = ResultSummarizer(
        args.clf, args.tackle_detector, args.pose_detector, 
        args.ball_detector, args.tackle_classifier)
    for video_file in tqdm(video_files):
        _result, vals = summarizer.summarize(video_file)
        result.append(_result)
        gts.append(vals[0])
        preds.append(vals[1])

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    u_score = utils.eval_score_tot(gts, preds)
    
    df = pd.DataFrame(result, columns=["video", "is_hia", "hia_pred", "score"])
    df.to_csv(summarizer.save_loc + "/scores.csv")
    print(summarizer.save_loc)

    gt = df.loc[:, "is_hia"].values
    pred = df.loc[:, "hia_pred"].values
    tpr = (gt * pred).sum() / gt.sum()
    result_text = (
        f"Utility score: {u_score: .4f}\n"
        f"True Positive Rate: {tpr:.3f}\n"
        f'Frame score {df.loc[:, "score"].mean():.3f}'
    )
    print(result_text)
    with open(summarizer.save_loc + "/summary.txt", "w") as f:
        f.write(result_text)
