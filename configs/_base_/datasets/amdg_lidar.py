# configs/_base_/datasets/amdg_lidar.py
dataset_type = 'Det3DDataset'
data_root = '../../data/amdg/test_mmdet3d_gen'
class_names = ['car']
metainfo = dict(classes=class_names)

# High-speed AV-24 racing envelope: X [-80m, +100m], Y [-25m, +25m], Z [-2m, +4m]
point_cloud_range = [-80.0, -25.0, -2.0, 100.0, 25.0, 4.0]

# 6 input channels: [x, y, z, reflectance, line_index, sensor_id]
# use_dim=[0, 1, 2, 3] passes x, y, z, and reflectance to the backbone
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3]
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='PointShuffle'
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3]
    ),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points']
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='amdg_infos_train.pkl',
        data_prefix=dict(pts=''),
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='amdg_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

test_dataloader = val_dataloader