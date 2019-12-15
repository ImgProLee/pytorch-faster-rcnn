# 将对象检测候选分配给ground truth目标。生成候选分类标签和边界框回归目标。为选择出的rois找到训练所需的ground truth类别和坐标变换信息

import torch
import torch.nn as nn
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground_truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._nm_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
