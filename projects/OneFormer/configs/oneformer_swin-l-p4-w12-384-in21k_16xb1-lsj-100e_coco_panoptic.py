_base_ = [
    './_base_/oneformer_swin-l-p4-w12-384-in21k-lsj_panoptic.py', 
    'mmdet::_base_/datasets/coco_panoptic.py'
    ]

model = dict(
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        ),
)
backend_args=None
dataset_type = 'CocoDataset'
data_root='data/coco/'
test_pipeline = [
    dict(type='LoadImageFromFile',
         imdecode_backend='pillow', 
         backend_args=backend_args),
    dict(
        type='ResizeShortestEdge', scale=800, max_size=1333, backend='pillow'),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        backend_args=backend_args),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
        backend_args=backend_args)
]
test_evaluator = val_evaluator