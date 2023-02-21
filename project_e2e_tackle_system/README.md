# High risk tackle detecting system

Directories in `src/`.

1. `dev`

Contains code to run CenterTrack.
(CenterTrack requires different environment than other models.)

2. `eval_component`

Contains code to evluate trained tackle frame selection model and tackle risk classification model.

3. `hia_detector`

Contains code to apply tackle frame selection model, tackle detection model, pose estimation model and tackle risk classification model.

4. `model_prep`

Contains code to train tackle frame selection model (`video_classifier/`) and tackle detection model (`tackle_detector/`).

To train model, `python run_train.py` at each directory.

## Citation

End-to-End High-Risk Tackle Detection System for Rugby

```
@inproceedings{nonaka2022end,
  title={End-to-End High-Risk Tackle Detection System for Rugby},
  author={Nonaka, Naoki and Fujihira, Ryo and Nishio, Monami and Murakami, Hidetaka and Tajima, Takuya and Yamada, Mutsuo and Maeda, Akira and Seita, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3550--3559},
  year={2022}
}
```