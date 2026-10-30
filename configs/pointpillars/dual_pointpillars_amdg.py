# configs/pointpillars/dual_pointpillars_amdg.py
# usage: python tools/train.py configs/pointpillars/dual_pointpillars_amdg.py
_base_ = [
    '../_base_/schedules/cyclic-40e.py',
    '../_base_/default_runtime.py'
]

# ==============================================================================
# Point Cloud Range & Grid Geometry
# [x_min, y_min, z_min, x_max, y_max, z_max]
# ==============================================================================
point_cloud_range = [-80.0, -25.0, -2.0, 100.0, 25.0, 4.0]
voxel_size = [0.25, 0.25, 6.0]
grid_shape = [200, 720]  # [Ny, Nx]

data_root = 'data/amdg/test_mmdet3d_gen/'
class_names = ['car']
metainfo = dict(classes=class_names, categories={'car': 0})
dataset_type = 'AMDGDataset'


# ==============================================================================
# Pipelines
# ==============================================================================
train_pipeline = [
    # 1. LiDAR Ingestion: [x, y, z, intensity]
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3]
    ),
    # 2. Radar Ingestion: [x, y, z, v_r, rcs, snr]
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=10,
        use_dim=[0, 1, 2, 3, 4, 5]
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'radar_points', 'gt_bboxes_3d', 'gt_labels_3d']
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
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=10,
        use_dim=[0, 1, 2, 3, 4, 5]
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'radar_points'])
]

# ==============================================================================
# Model Architecture Updates (Dual-Branch LiDAR + Radar)
# ==============================================================================
model = dict(
    type='DualBranchPointPillars',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False  # Handled per-branch inside the detector
    ),
    # --- Branch A: Dense LiDAR Stream ---
    lidar_voxel_layer=dict(
        type='HardVoxelize',
        max_num_points=32,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)
    ),
    lidar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,          # [x, y, z, intensity]
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    ),
    lidar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=grid_shape
    ),
    # --- Branch B: Sparse Radar Stream ---
    radar_voxel_layer=dict(
        type='HardVoxelize',
        max_num_points=16,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(4000, 8000)
    ),
    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=6,          # [x, y, z, v_r, rcs, snr]
        feat_channels=[32],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=32,
        output_shape=grid_shape
    ),
    # --- Shared 2D Backbone & Detection Head ---
    backbone=dict(
        type='SECOND',
        in_channels=96,         # 64 (LiDAR BEV) + 32 (Radar BEV)
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-80.0, -25.0, 0.42, 100.0, 25.0, 0.42]],
            sizes=[[4.876, 1.93, 1.50]],
            rotations=[0, 1.57],
            reshape_out=False
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2
        ),
        # Pass test_cfg directly to the head so it is never None during predict()
        test_cfg=dict(
            use_rotate_nms=True,
            use_raw_score=False,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50
        )
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1
            )
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        use_raw_score=False,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50
    )
)

# ==============================================================================
# DataLoader Overrides
# ==============================================================================
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
        modality=dict(use_lidar=True, use_radar=True, use_camera=False),
        metainfo=metainfo,
        box_type_3d='LiDAR',
        filter_empty_gt=False
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
        modality=dict(use_lidar=True, use_radar=True, use_camera=False),
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False
    )
)

test_dataloader = val_dataloader

# ==============================================================================
# Evaluator Definitions
# ==============================================================================
val_evaluator = dict(
    type='AMDGMetric',
    iou_thrs=[0.25, 0.5, 0.7],
    prefix='AMDG'
)
test_evaluator = val_evaluator

# ==============================================================================
# Default Hooks & Runtime
# ==============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook', draw=False)
)

train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

work_dir = './work_dirs/dual_pointpillars_amdg'