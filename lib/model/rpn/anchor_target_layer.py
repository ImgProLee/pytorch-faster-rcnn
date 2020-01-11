

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from lib.model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False
try:
    long        # Python 2
except NameError:
    long = int  # Python 3
class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]   # [1, 18, 36, 57]  Softmax前的卷积层输出
        gt_boxes = input[1]   # [1, 20, 5]
        im_info = input[2]   # [1, 3]
        num_boxes = input[3]  # [1]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        batch_size = gt_boxes.size(0)
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 从坐标向量中返回坐标矩阵,元素交叉
        # torch.from_numpy将np数据转化为tensor，将tensor转化为np：tensor.numpy()
        # ravel()函数与flatten()函数功能类似，将多维数组降一维，np.flatten返回拷贝，不会影响原始数据，np.ravel返回视图view
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        # shift_x,shift_y为[37,53]矩阵，展平后堆叠再转置，得到[1961,3]tensor
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()  # contiguous()把tensor变为连续分布形式

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.  [9, 4]
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)   # [17649, 4]

        total_anchors = int(K * A)  # e.g. 17649
        # 限制anchor不超出图像边界
        keep = ((all_anchors[:, 0] >= -self._allowed_border) & \
               (all_anchors[:, 1] >= -self._allowed_border) & \
               (all_anchors[:, 2] < long(im_info[0][1]) +self._allowed_border) & \
               (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))
        # len(keep) = 19647
        # 选择keep不为0的anchor，即选择在范围内的anchor.
        # 一般过滤后只会有原来1/3左右的anchor。如果不将这部分anchor过滤，则会使训练过程难以收敛。
        inds_inside = torch.nonzero(keep).view(-1)  # torch.Size([7528])
        anchors = all_anchors[inds_inside, :]

        # label: 1 is postive, 0 is negative, -1 is don't care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)  # [1, 7528]
        # bbox_inside_weights用来设置正样本回归loss的权重，type与gt_boxes保持一致
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()  # [1, 7528]
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_() # [1, 7528]

        # 计算所有anchor与gt_boxes的交并比，返回的交并比数量为 N * K
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)  # [7528, 4] [1, 20, 5]
        # overlaps = [1, 7528, 20]表示所有anchor和box的交并比，每一行表示一个anchor与20个gt_box的交并比

        # max_overlaps [1, 7528]表示所有anchor与所有gt_box的交并比最大值，argmax_overlaps表示每个anchor与gt box取到最大值的索引
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # gt_max_overlaps [1, 20]表示与每个gt_box最大交并比的anchor，
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # True  将IOU小于0.3的样本作为负样本
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5  # 1*10^-5
        # overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps))：表示overlaps中gt box和哪个anchor的IOU最大，
        # 忽略batch，[7528,20]矩阵，按列比较最大值，最大值相同那么其值就置为1， 其他的都置为0。这时会出现某些行全是0的情况，
        # 也就是20个gt box对应的anchor行为1，其他anchor所在行全为0，再按行求和，则得出非0行。
        # 表示这个anchor是与gt boxes具有最大IOU的anchor中的一个
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)  # [1, 7528]

        if torch.sum(keep) > 0:
            labels[keep>0] = 1  # 將与20个gt boxes具有最大IOU的anchor设置为正样本

        # fg label: above threshold IOU  IOU大于0.7的anchor设置为正样本
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # False
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 0.5*256=128个前景框
        sum_fg = torch.sum((labels == 1).int(), 1)  # 符合IOU>0.7的正样本数量
        sum_bg = torch.sum((labels == 0).int(), 1)  # 背景负样本数量

        for i in range(batch_size):
            # subsample postive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)  # 返回正样本所在的索引
                # 随机选择sum_fg - num_fg个anchor，将其label设置为-1，-1表示无效的anchor，不表示负样本
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]  # 256-正样本数量

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                # 随机选择一部分背景景(sum_bg-num_bg个)，置为-1(-1为无效框，不是背景框)，只保留num_bg个背景
                # 正负样本总共256个
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)  # 0
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)表示选出与每个anchor最大IOU的gt box
        # bbox_targets [1, 7528, 4] 表示每个anchor与其最大IOU的gt box的平移和缩放比例
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index. TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        # 设置正样本的bbox_inside_weights为1.0
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:   # -1
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights  # 1 / N
        bbox_outside_weights[labels == 0] = negative_weights  # 1 / N

        # 将label、labels、bbox_inside_weights、bbox_outside_weights分别扩充到原图的所有anchor中
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)  # [1,17649]
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)  # [1,17649]

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()  # [1,36,57,9]->[1,9,36,57]
        labels = labels.view(batch_size, 1, A * height, width)  # [1, 1, 9*36, 57]
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()  # [1,36,57,9*4]->[1,9*4,36,57]
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)  # 17649
        # [1,17649,1]->[1,17649,4]将正样本权重扩展到box的每个值上
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                        4)
        # [1, 17649, 4]->[1, 36,57,4*9]->[1, 4*9, 36, 57]
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                              4)
        # [1, 36, 57, 4*9]->[1, 4*9, 36, 57]
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs



def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret



def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


