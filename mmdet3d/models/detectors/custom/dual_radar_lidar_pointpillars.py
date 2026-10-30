import torch
from torch import nn
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.utils import OptConfigType, OptMultiConfig

from mmdet3d.models.task_modules.voxel import VoxelGenerator
from mmdet3d.registry import TASK_UTILS

from mmcv.ops import Voxelization

import pdb


@MODELS.register_module()
class DualBranchPointPillars(SingleStage3DDetector):
    """Dual-Branch PointPillars for synchronized LiDAR and Radar streams."""

    def __init__(self,
                 lidar_voxel_layer: OptConfigType = None,
                 radar_voxel_layer: OptConfigType = None,
                 lidar_voxel_encoder: OptConfigType = None,
                 radar_voxel_encoder: OptConfigType = None,
                 lidar_middle_encoder: OptConfigType = None,
                 radar_middle_encoder: OptConfigType = None,
                 backbone: OptConfigType = None,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)



        # 1. LiDAR Stream Modules
        # Build LiDAR voxel layer

        # Strip any extraneous 'type' key and build directly via mmcv.ops.Voxelization
        if isinstance(lidar_voxel_layer, dict):
            l_cfg = {k: v for k, v in lidar_voxel_layer.items() if k != 'type'}
            self.lidar_voxel_layer = Voxelization(**l_cfg)
        else:
            self.lidar_voxel_layer = lidar_voxel_layer

        self.lidar_voxel_encoder = MODELS.build(lidar_voxel_encoder)
        self.lidar_middle_encoder = MODELS.build(lidar_middle_encoder)



        # 2. Radar Stream Modules
        # Build Radar voxel layer

        if isinstance(radar_voxel_layer, dict):
            r_cfg = {k: v for k, v in radar_voxel_layer.items() if k != 'type'}
            self.radar_voxel_layer = Voxelization(**r_cfg)
        else:
            self.radar_voxel_layer = radar_voxel_layer

        self.radar_voxel_encoder = MODELS.build(radar_voxel_encoder)
        self.radar_middle_encoder = MODELS.build(radar_middle_encoder)

    def predict(self, batch_inputs_dict: dict, batch_data_samples: list, **kwargs):
        """Pass test_cfg to head if not present."""
        if getattr(self.bbox_head, 'test_cfg', None) is None:
            self.bbox_head.test_cfg = self.test_cfg
        return super().predict(batch_inputs_dict, batch_data_samples, **kwargs)
    
    def extract_feat(self, batch_inputs_dict: dict) -> tuple:
        """Extracts features from both modalities and fuses them in BEV space."""
        lidar_pts = batch_inputs_dict['points']
        radar_pts = batch_inputs_dict['radar_points']

        # --- A. LiDAR Branch ---
        l_voxels, l_num_points, l_coors = [], [], []
        for res in lidar_pts:
            pts_tensor = res.tensor if hasattr(res, 'tensor') else res
            v, c, n = self.lidar_voxel_layer(pts_tensor)
            l_voxels.append(v)
            l_coors.append(c)
            l_num_points.append(n)

        l_voxels = torch.cat(l_voxels, dim=0)
        l_num_points = torch.cat(l_num_points, dim=0)
        l_coors_batch = []
        for i, c in enumerate(l_coors):
            c_pad = nn.functional.pad(c, (1, 0), mode='constant', value=i)
            l_coors_batch.append(c_pad)
        l_coors = torch.cat(l_coors_batch, dim=0)

        l_pillar_feats = self.lidar_voxel_encoder(l_voxels, l_num_points, l_coors)
        l_bev = self.lidar_middle_encoder(l_pillar_feats, l_coors, len(lidar_pts))

        # --- B. Radar Branch ---
        r_voxels, r_num_points, r_coors = [], [], []
        for res in radar_pts:
            pts_tensor = res.tensor if hasattr(res, 'tensor') else res
            if pts_tensor.shape[0] > 0:
                v, c, n = self.radar_voxel_layer(pts_tensor)
            else:
                # Handle empty radar scans gracefully without CUDA crashes
                device = pts_tensor.device
                feat_dim = pts_tensor.shape[-1]
                v = torch.zeros((1, self.radar_voxel_layer.max_num_points, feat_dim), device=device)
                c = torch.zeros((1, 3), dtype=torch.long, device=device)
                n = torch.ones((1,), dtype=torch.int32, device=device)
            r_voxels.append(v)
            r_coors.append(c)
            r_num_points.append(n)

        r_voxels = torch.cat(r_voxels, dim=0)
        r_num_points = torch.cat(r_num_points, dim=0)
        r_coors_batch = []
        for i, c in enumerate(r_coors):
            c_pad = nn.functional.pad(c, (1, 0), mode='constant', value=i)
            r_coors_batch.append(c_pad)
        r_coors = torch.cat(r_coors_batch, dim=0)

        r_pillar_feats = self.radar_voxel_encoder(r_voxels, r_num_points, r_coors)
        r_bev = self.radar_middle_encoder(r_pillar_feats, r_coors, len(radar_pts))

        # --- C. Fused BEV Map ---
        fused_bev = torch.cat([l_bev, r_bev], dim=1)
        x = self.backbone(fused_bev)
        if self.with_neck:
            x = self.neck(x)
        return x

 