from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes, bbox3d_mapping_back


@METRICS.register_module()
class AMDGMetric(BaseMetric):
    """Evaluation metric for AMDG LiDAR-based 3D detection.
    
    Computes 3D and BEV Average Precision (AP) directly from LiDAR bounding boxes
    without camera projection matrices or image plane clipping.
    """

    def __init__(self,
                 ann_file: Optional[str] = None,
                 iou_thrs: Sequence[float] = (0.25, 0.5, 0.7),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.iou_thrs = iou_thrs

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of predictions and ground truths."""
        for sample in data_samples:
            # Extract ground truths
            eval_ann_info = sample.get('eval_ann_info', {})
            if not eval_ann_info:
                gt_instances = sample.get('gt_instances_3d', {})
                gt_bboxes = gt_instances.get('bboxes_3d', None)
                gt_labels = gt_instances.get('labels_3d', None)
            else:
                gt_bboxes = eval_ann_info.get('gt_bboxes_3d', None)
                gt_labels = eval_ann_info.get('gt_labels_3d', None)

            # Extract predictions
            pred_instances = sample.get('pred_instances_3d', {})
            pred_bboxes = pred_instances.get('bboxes_3d', None)
            pred_scores = pred_instances.get('scores_3d', None)
            pred_labels = pred_instances.get('labels_3d', None)

            # Store on CPU as numpy arrays
            gt_dict = {
                'bboxes': gt_bboxes.tensor.cpu().numpy() if hasattr(gt_bboxes, 'tensor') else np.zeros((0, 7)),
                'labels': gt_labels.cpu().numpy() if isinstance(gt_labels, torch.Tensor) else np.zeros((0,))
            }

            pred_dict = {
                'bboxes': pred_bboxes.tensor.cpu().numpy() if hasattr(pred_bboxes, 'tensor') else np.zeros((0, 7)),
                'scores': pred_scores.cpu().numpy() if isinstance(pred_scores, torch.Tensor) else np.zeros((0,)),
                'labels': pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else np.zeros((0,))
            }

            self.results.append((pred_dict, gt_dict))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute mAP at configured IoU thresholds."""
        eval_results = {}
        for iou_thr in self.iou_thrs:
            tp_list = []
            fp_list = []
            scores_list = []
            num_gts = 0

            for pred_dict, gt_dict in results:
                preds = pred_dict['bboxes']
                scores = pred_dict['scores']
                gts = gt_dict['bboxes']

                num_gts += len(gts)
                if len(preds) == 0:
                    continue

                if len(gts) == 0:
                    for s in scores:
                        tp_list.append(0)
                        fp_list.append(1)
                        scores_list.append(s)
                    continue

                # Calculate 3D IoU using LiDARInstance3DBoxes
                p_box = LiDARInstance3DBoxes(preds)
                g_box = LiDARInstance3DBoxes(gts)
                overlaps = p_box.overlaps(p_box, g_box).numpy()

                # Sort by detection confidence
                sort_inds = np.argsort(-scores)
                assigned_gt = set()

                for p_idx in sort_inds:
                    max_iou = -1.0
                    match_g = -1
                    for g_idx in range(len(gts)):
                        if g_idx in assigned_gt:
                            continue
                        iou = overlaps[p_idx, g_idx]
                        if iou > max_iou:
                            max_iou = iou
                            match_g = g_idx

                    scores_list.append(scores[p_idx])
                    if max_iou >= iou_thr and match_g != -1:
                        tp_list.append(1)
                        fp_list.append(0)
                        assigned_gt.add(match_g)
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

            if len(scores_list) == 0:
                eval_results[f'mAP_3d@{iou_thr}'] = 0.0
                continue

            scores_arr = np.array(scores_list)
            tp_arr = np.array(tp_list)
            fp_arr = np.array(fp_list)

            sort_inds = np.argsort(-scores_arr)
            tp_arr = tp_arr[sort_inds]
            fp_arr = fp_arr[sort_inds]

            cum_tp = np.cumsum(tp_arr)
            cum_fp = np.cumsum(fp_arr)

            recall = cum_tp / max(num_gts, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

            # 11-point interpolated AP
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_interp = np.max(precision[recall >= t]) if np.sum(recall >= t) > 0 else 0
                ap += p_interp / 11.0

            eval_results[f'mAP_3d@{iou_thr}'] = float(ap)
            print_log(f'AMDG Evaluation: 3D mAP @ IoU={iou_thr}: {ap:.4f} (GTs: {num_gts})', logger='current')

        return eval_results