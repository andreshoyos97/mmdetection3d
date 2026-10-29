# mmdet3d/datasets/amdg_dataset.py
from typing import Callable, List, Optional, Union
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class AMDGDataset(Det3DDataset):
    r"""AMDG Dataset for Autonomous Racing LiDAR 3D Detection."""

    METAINFO = {
        'classes': ('car',),
        'categories': {'car': 0}
    }

    def __init__(self,
                 data_root: str = '',
                 ann_file: str = '',
                 pipeline: Optional[List[Union[dict, Callable]]] = None,
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Standardize point cloud paths and instance metadata."""
        info = super().parse_data_info(raw_data_info)

        # Map nested point cloud path if present
        if 'lidar_points' in info and 'lidar_path' in info['lidar_points']:
            info['lidar_path'] = info['lidar_points']['lidar_path']
            info['num_pts_feats'] = info['lidar_points'].get('num_pts_feats', 4)

        # Synchronize labels for dataset logging and summary statistics
        for inst in info.get('instances', []):
            if 'bbox_label' not in inst:
                inst['bbox_label'] = inst.get('bbox_label_3d', 0)
            if 'name' not in inst:
                inst['name'] = 'car'

        ann_info = self.parse_ann_info(info)
        info['ann_info'] = ann_info
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Wrap raw bounding box numpy arrays into LiDARInstance3DBoxes."""
        ann_info = dict()
        instances = info.get('instances', [])

        gt_bboxes_3d = []
        gt_labels_3d = []

        for inst in instances:
            # Skip invalid ground truths if flagged
            if not inst.get('bbox_3d_isvalid', True):
                continue

            gt_bboxes_3d.append(inst['bbox_3d'])
            label = inst.get('bbox_label_3d', inst.get('bbox_label', 0))
            if isinstance(label, str):
                label = self.METAINFO['classes'].index(label)
            gt_labels_3d.append(label)

        if len(gt_bboxes_3d) > 0:
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)

        # Convert to MMDetection3D spatial box structure
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1]
        )
        ann_info['gt_labels_3d'] = gt_labels_3d

        return ann_info

    def get_cat_ids(self, idx: int) -> list:
        """Return category IDs for the instance summary table and class sampling."""
        data_info = self.get_data_info(idx)
        instances = data_info.get('instances', [])
        cat_ids = []
        for inst in instances:
            if inst.get('bbox_3d_isvalid', True):
                cat_ids.append(inst.get('bbox_label_3d', inst.get('bbox_label', 0)))
        return cat_ids