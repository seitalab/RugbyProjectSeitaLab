# The new config inherits a base config to highlight the necessary modification
_base_ = 'retinanet_r50_fpn_mstrain_3x_coco_ball_base.py'

mode = "/base"
DATA_LOC = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/ball/processed" + mode

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('ball',)
data = dict(
    train=dict(
        img_prefix=DATA_LOC+'/train/',
        classes=classes,
        ann_file=DATA_LOC+'/train/annotation_coco.json'),
    val=dict(
        img_prefix=DATA_LOC+'/valid/',
        classes=classes,
        ann_file=DATA_LOC+'/valid/annotation_coco.json'),
    # test=dict(
    #     img_prefix=DATA_LOC+'/test/',
    #     classes=classes,
    #     ann_file=DATA_LOC+'/test/annotation_coco.json')
    )
