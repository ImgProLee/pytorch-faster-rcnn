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
        # 为了便于softmax分类,[1,18,36,57]到[1,2,36*9,57],宽度为9999的堆叠得到324宽度，57高度
        # 2深度，对所有宽高对应的2个值求softmax，再reshape回去，18对应9个anchor连续前景得分+连续背景得分
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        # _ProposalLayer 传入预测的rpn_box得分和坐标修正值，图像im_info，
        # 在ProposalLayer中特征图每个位置生成anchor，对anchor进行修正，排序，裁剪等处理。
        # 只返回rois，并不包含任何loss，loss只在训练过程中使用，通过_AnchorTargetLayer计算
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and bulid the rpn loss
        if self.training:
            assert gt_boxes is not None

            # _AnchorTargetLayer传入rpn_box得分预测，gt_box以及im_info，num_boxes等信息。_AnchorTargetLayer中同样在特征图每个位置生成anchor，
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss  [1, 2, 36*9, 57]->[1, 36*9, 57, 2]->[1, 36*9*57, 2]
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)  # [1, 17649]
            # torch.ne(input, other, out=Tensor) -> Tensor 如果tensor != other 为True，返回1
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))  # 通过ne去掉-1，返回非0索引[17649]，索引中包含所有正负样本的索引
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)  # 根据索引选出在[17649,2]的0维度选择score
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep)  # 同样根据索引选择label
            rpn_label = Variable(rpn_label.long())
            # 损失函数，rpn只负责预测anchor是前景还是背景，因此只有二分类
            # [b*9*w*h, 2] 和[b*9*w*h]  由于使用的交叉熵损失函数，里面会计算Softmax。
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_indide_weights, rpn_bbox_outside_weights = rpn_data[1:] # 取anchor_target_layer的返回值

            # compute bbox regression loss
            rpn_bbox_indide_weights = Variable(rpn_bbox_indide_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_indide_weights, rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])


        return rois, self.rpn_loss_cls, self.rpn_loss_box


