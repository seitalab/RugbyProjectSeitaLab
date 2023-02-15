# The new config inherits a base config to highlight the necessary modification
_base_ = 'retina_base/retinanet_r50_caffe_fpn_mstrain_3x_coco.py'

# DATA_LOC = "/export/work/data/osaka_rugby/work/nonaka/tackle_detection/yolo_finetune_v211115/tackle_only"

model = dict(
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('tackle',)
# data = dict(
#     train=dict(
#         img_prefix=DATA_LOC+'/train/',
#         classes=classes,
#         ann_file=DATA_LOC+'/train/annotation_coco.json'),
#     val=dict(
#         img_prefix=DATA_LOC+'/valid/',
#         classes=classes,
#         ann_file=DATA_LOC+'/valid/annotation_coco.json'),
#     # test=dict(
#     #     img_prefix=DATA_LOC+'/test/',
#     #     classes=classes,
#     #     ann_file=DATA_LOC+'/test/annotation_coco.json')
#     )

# We can use the pre-trained model to obtain higher performance
load_from = 'checkpoints/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'
