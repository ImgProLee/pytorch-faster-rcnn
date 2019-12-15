
import torch
import torch.nn as nn
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.generate_anchors import generate_anchors
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.model.nms.nms_wrapper import nms

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    def __init__(self, feat_strid, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_strid
        self._anchors = torch.from_numpy(generate_anchors(scales = np.array(scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        # self._anchors返回值为  [9，4]9个anchor坐标

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        # input输入形式为tuple = (rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)
        # 0-8为anchor的背景得分, 9-17为anchor的前景得分
        scores = input[0][:, self._num_anchors:, :, :]  # [1,9,53,37]
        bbox_deltas = input[1]  # [1,36,63,37]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # nms之前保存的建议区域数量，检测阶段为6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 通过nms后保存的建议区域数量，检测阶段为300
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # ms的阈值 检测阶段0.7
        min_size = cfg[cfg_key].RPN_MIN_SIZE  # 建议区域的最小宽度或高度，检测阶段为16

        batch_size = bbox_deltas.size(0)  # batch_size = 1

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 从坐标向量中返回坐标矩阵,元素交叉
        # torch.from_numpy将np数据转化为tensor，将rensor转化为np：tensor.numpy()
        # ravel()函数与flatten()函数功能类似，将多维数组降一维，np.flatten返回拷贝，不会影响原始数据，np.ravel返回视图view
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        # shift_x,shift_y为[37,57]矩阵，展平后堆叠再转置，得到[1961,3]tensor
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()  #contiguous()把tensor变为连续分布形式

        A = self._num_anchors
        K = shifts.size(0)

        # 9个anchor，每个包含四个坐标偏移值，宽高中心点坐标
        self._anchors = self._anchors.type_as(scores)  # [9,4]
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)  # [1961, 9, 4]
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)  # [1, 17649, 4]

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchor

        bbox_deltas = bbox_deltas.permute(0,2,3,1).contiguous()  # [1, 63, 37, 36]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # [1, 17649, 4]

        # Same story for the score
        scores = scores.permute(0,2,3,1).contiguous()  # permute将维度换位
        scores = scores.view(batch_size, -1)    # [1, 17649]

        # Convert anchors into proposals via bbox transformations
        # 根据anchor和偏移量计算proposals,delta表示偏移量，返回左上和右下顶点的坐标(x1,y1,x2,y2)
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # clip predicted boxes to image，将proposals限制在图片范围内，超出边界，则将边界赋值
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)  # _ is scores after sort,order is index after scores

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            # 从[1,17949,4]转换到[17649,4],从[1, 17649]转换到[17649]
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            # numel函数返回元素个数
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]  # 测试阶段取前6000个得分的索引

            # 取前300的索引对应的区域和得分,[6000,4],[6000,1]，这里会重新生成proposals_single的下标0：5999
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            # torch.cat 在第1维度拼接区域和得分矩阵，[6000,5]
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            # keep_idx_i 返回通过nms阈值限制之后的索引，该索引基于6000的下标[102,1]或[561,1]
            keep_idx_i = keep_idx_i.long().view(-1)

            # 取该索引的前300个建议区域
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.，将不足300的建议区域补0
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep


