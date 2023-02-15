# The new config inherits a base config to highlight the necessary modification
_base_ = 'retinanet_r50_fpn_mstrain_3x_coco_tackle_base.py'

random_frame_ratio = 10
num_ratio = f'/random_ratio_{random_frame_ratio:02d}'
DATA_LOC = "/export/work/data/osaka_rugby/work/nonaka/hia_detect_system/tackle_detector/tackle_frame_image" + num_ratio


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('tackle',)
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
