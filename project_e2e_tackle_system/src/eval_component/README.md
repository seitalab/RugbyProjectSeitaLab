
- `apply_video_classifier.py`
    Apply classifier to evaluate labeled frames in test set.

- `pose_clf.py`
    Apply tackle risk classification model and store pairs of posture evaluated as high risk tackle.
    Classification applied to all pairs and store result for each pair.

- `tackle_clf.py`
    Apply tackle risk classification model and store result.
    Classification applied to all pairs, only store binary value (high risk tackle existed or not).
    (Evaluation target includes, `pose-detector` with humancheck option.)