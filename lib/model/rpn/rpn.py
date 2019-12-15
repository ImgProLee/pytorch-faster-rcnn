import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss

class _RPN(nn.Module):
    """region proposal network"""
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din  #get depth of the feature map e.g.,512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        # 特征步长 __C.FEAT_STRIDE = [16, ]，记录图像经过特征图缩小的尺度
        self.feat_stride = cfg.FEAT_STRIDE[0]

        #define the convrelu layers processing input feature map
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # 对每个anchor都要进行背景或前景的分类得分，个数就是尺度个数乘以比例个数再乘以2
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 偏移的输出个数是anchor个数乘以4, 每个anchor4个坐标偏移值
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer定义候选区域层,
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer将anchor对应ground truth。生成anchor分类标签和边界框回归目标。
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod     #静态方法，可以不需要实例化，直接类名.方法名()来调用。
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        #get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # 为了便于softmax分类,[1,18,36,57]到[1,2,36*9,57],宽度为999的堆叠得到324宽度，57高度
        # 2深度，对所有宽高对应的2个值求softmax，再reshape回去，18对应9个anchor连续前景得分+连续背景得分
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        return rois, self.rpn_loss_cls, self.rpn_loss_box


