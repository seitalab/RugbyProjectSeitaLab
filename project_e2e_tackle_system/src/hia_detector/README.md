# Procedure.

1. `01_apply_video_classifier.py`

2. `02a_apply_detector_to_video.py`

3. `02b_pose_detector.py`
    To apply CenterTrack, need to run `run_centertrack.py` at `src/dev`.

4. `03_tackle_pose_extractor.py`

5. `04_apply_classifier_to_pose.py`

6. `05_summarize_detection_result.py`


- `bulk_exe.py`
    Used to execute tackle_pose_extraction, pose classification and score summarization.

- `bulk_exe_tackleclf.py`
    Used to execute tackle classification with various settings at once.

- `bulk_exe_summarize.py`
    Used to execute result summarization (= score calculation) at once for various settings.

- `05b_summarize_detection_result_atira.py`
    Used to evaluate data of `Automated Tackle Injury Risk Assessment in Contact-Based Sports - A Rugby Union Example`.
    ```
    @inproceedings{martin2021automated,
        title={Automated tackle injury risk assessment in contact-based sports-a rugby union example},
        author={Martin, Zubair and Hendricks, Sharief and Patel, Amir},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={4594--4603},
        year={2021}
    }
    ```
    Recommended to change directory in config.yaml.

- Files required by each model is defined in `resource/model_info.yaml`