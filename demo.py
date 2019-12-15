#------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import _init_path
import os
import sys
import numpy as np
import argparse    #参数传递
import pprint     #提供了打印出任何python数据结构类和方法。
import pdb   #python调试器
import time
import cv2   #计算机视觉库
#import imutils   #在opencv基础上对一些方法进行了再次加工，简单易用
import torch
from torch.autograd import Variable  #创建pytorch变量，可调用backward()方法计算梯度，通过data属性访问张量。
import torch.nn as nn    #函数包装库
import torch.optim as optim  #是一个实现了各种优化算法的库

import torchvision.transforms as transforms  #包含图像裁剪缩放等数据增强函数
import torchvision.datasets as dset    #包含MNIST、ImageNet、COCO等常用数据集，并提供参数设置，进行数据调用
from scipy.misc import imread    #图像相关的io操作
# from roi_data_layer.roidb import combined_roidb
# from roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file   #参数配置对象train、test……
from lib.model.utils.blob import im_list_to_blob
from lib.model.faster_rcnn import vgg16
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv



def parse_args():
    #创建命令行选项、参数和子命令解析器
    parse = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    #给解析器添加程序参数，name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo
    parse.add_argument('--dataset', dest='dataset',help='training dataset', default='pascal_voc',type=str)
    #dest - 被添加到 parse_args() 所返回对象上的属性名。例：args = parse_args(),args.dest
    parse.add_argument('--cfg', dest='cfg_file',help='optional config file',default='cfgs/vgg16.yml',type=str)
    #default - 当参数未在命令行中出现时使用的值。
    parse.add_argument('--net',dest='net',help='vgg16, res50, res101',default='vgg16',type=str)
    #type - 命令行参数应当被转换成的类型。
    parse.add_argument('--set',dest='set_cfgs',help='set config keys', default=None,nargs=argparse.REMAINDER)
    parse.add_argument('--load_dir',dest='load_dir',help='directory to load model',default='models')
    #parse.add_argument('--set',dest='set_cfgs',help='set config keys',default=None)
    #action - 当参数在命令行中出现时使用的动作基本类型。action='store_true'，只要运行时该变量有传参就将该变量设为True。
    parse.add_argument('--cuda',dest='cuda',help='whether use cuda',action='store_true')
    #required - 此命令行选项是否可省略。
    parse.add_argument('--image_dir',dest='image_dir',help='directory to load images for demo',default='images')
    parse.add_argument('--parallel_type',dest='parallel_type',help='which part of model to parallel,0:all,1:model before roi pooling',default=0,type=int)
    #加载训练好的faster rcnn模型指定保存的模型参数
    parse.add_argument('--checksession', dest='checksession', help='checksession to load model',default=1,type=int)
    parse.add_argument('--checkepoch',dest='checkepoch',help='checkepoch to load model',default=1,type=int)
    parse.add_argument('--checkpoint',dest='checkpoint',help='checkpoint to load model',default=10021,type=int)
    parse.add_argument('--bs',dest='batch_size',help='batch_size',default=1,type=int)

    args = parse.parse_args()
    return args

def _get_image_blob(im):
    """将一幅图像转化为网络需要的输入
    Arguments:输入一个通道顺序为BGR的图像
      im (ndarray): a color image in BGR order
    Returns:返回一个图像金字塔列表
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    #减去数据训练集的统计平均值，来消除公共的部分，以凸显个体之间的特征和差异
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        # 限制最小边为600，最大边为1000，对于输入图像优先考虑最大边的限制
        # 输入图像的大小是：375*500*3，则resize后的图像大小为：600*800*3
        # 输入图像的大小是：375*800*3，则resize后的图像大小为：469*1000*3
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:   #np.round返回四舍五入值
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        #resize参数：输入图像、输出图像、输出尺寸、w方向缩放因子、h方向…、插值方法
        im = cv2.resize(im_orig, None, None, fx=im_scale,fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    #Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
       cfg_from_file(args.cfg_file)  #将yml文件中的网络参数合并到cfg中
    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    # if not os.path.exists(input_dir):
    #     raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    #np.asarray不对目标做拷贝，不占用内存
    pascal_classes = np.asarray(['__background__',
                               'aeroplane', 'bicycle','bird', 'boat', 'bottle',
                               'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog',
                               'horse', 'motorbike','person', 'pottedplant', 'sheep',
                               'sofa', 'train', 'tvmonitor'])
    # initlize the network here
    fasterRCNN = vgg16(pascal_classes, pretrained = False, class_agnostic = args.class_agnostic)
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)   # 在GPU中加载模型
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print("load model successfully!")
    # initilize the tensor holder here
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():   # 设置无需反向传播
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()
    # eval（）时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    # model.train()启用BatchNormalization 和 Dropout
    fasterRCNN.eval()
    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num

    #set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        imglist = os.listdir(args.image_dir)
        num_images = len(imglist)
    print('Load Poto: {} images.' .format(num_images))

    while (num_images >= 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            im_in = np.array(imread(im_file))
            if len(im_in.shape) == 2:
                im_in = im_in[:,:,np.newaxis]
                im_in = np.concatenate((im_in, im_in, im_in), axis=2)
            # rgb ->bgr,[::-1]从后向前取相反的元素
            im_in = im_in[:,:,::-1]
        im = im_in
        blobs, im_scales = _get_image_blob(im)
        # blob[1,850,600,3], im_scales[1.6991]
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)  # shape(1,3)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0,3,1,2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas, bbox_pred的shape为[1,300,84]
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        # 计算结束bbox_pred的shape为[6300,4]
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
                    # 将box_deltas转为[1,300,84],便于修正rois的box

            # bbox为rpn网络生成的roi候选区域,box_delta为Faster rcnn全连接层生成的候选区域修正值
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
            pred_boxes = _.cuda() if args.cuda > 0 else _
        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxex = pred_boxes[inds, :]
                else:
                    cls_boxex = pred_boxes[inds][:, j * 4:(j+1)*4]

                cls_dets = torch.cat((cls_boxex, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                # TEST.NMS = 0.3
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu = not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()
        if vis and webcam_num == -1:
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            cv2.imshow("frame", im2show)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()


