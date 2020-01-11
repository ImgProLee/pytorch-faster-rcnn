
import os
import numpy as np
import argparse
import pdb
import pprint
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient
from lib.model.faster_rcnn.vgg16 import vgg16


def parse_args():
    #创建命令行选项、参数和子命令解析器
    parse = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    #给解析器添加程序参数，name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo
    parse.add_argument('--dataset', dest='dataset',help='training dataset', default='pascal_voc',type=str)
    parse.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parse.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=20, type=int)
    parse.add_argument('--net', dest='net', help='vgg16, res50, res101', default='vgg16', type=str)
    parse.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100, type=int)
    parse.add_argument('--save_dir', dest='save_dir', help='directory to save models', default='models', type=str)
    parse.add_argument('--nw', dest='num_workers', help='number of workers to load data', default=0, type=int)
    #dest - 被添加到 parse_args() 所返回对象上的属性名。例：args = parse_args(),args.dest
    parse.add_argument('--cfg', dest='cfg_file',help='optional config file',default='cfgs/vgg16.yml',type=str)
    parse.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parse.add_argument('--mGPUs', dest='mGPUs', help='whether to use multiple GPUs', action='store_true')
    parse.add_argument('--bs',dest='batch_size',help='batch_size',default=1,type=int)
    parse.add_argument('--cag', dest='class_agnostic', help='whether to perform class_agnostic bbox regression', action='store_true')
    parse.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parse.add_argument('--ls', dest='large_scale', help='whether use large imag scale', action='store_true')
    parse.add_argument('--lr', dest='lr', help='starting learning rate', default=0.001, type=float)
    parse.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch', default=5, type=int)
    parse.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    # set training session
    parse.add_argument('--s', dest='session', help='training session', default=1, type=int)
    parse.add_argument('--r', dest='resume', help='resume checkpoint or not', default=False, type=bool)
    parse.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parse.add_argument('--checkepoch',dest='checkepoch',help='checkepoch to load model',default=1,type=int)
    parse.add_argument('--checkpoint',dest='checkpoint',help='checkpoint to load model',default=23079,type=int)
    # log and display
    parse.add_argument('--use_tfb', dest='use_tfboard', help='whether use tensorboard', action='store_true')

    args = parse.parse_args()
    return args

# 采样器继承Sampler类
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()  # 0
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()  # []
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size  # 随机打乱
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)  # 返回一个索引迭代器

  def __len__(self):
    return self.num_data

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    args.imdb_name = "voc_2007_trainval"
    args.imdb_val_name = "voc_2007_test"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config: ")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    if torch.cuda.is_available() and not args.cuda:
        print("Warning: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # --Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # imdb 表示对应voc_2007_trainval的pascal voc对象
    # roidb表示对应图像的描述信息，包括图像本身属性以及boxes的信息
    # ratio_list表示所有训练图像的宽高比例从小到大的排序
    # ratio_index表示排序后的ratio_list对应image_index的索引
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    # output_dir = models/vgg16/pascal_voc
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Sampler生成索引迭代器，DataLoader读取数据时使用
    sampler_batch = sampler(train_size, args.batch_size) # 23080 1
    # roibatchLoader继承data.Dataset类，构建DataLoader可读取的数据集
    dataset = roibatchLoader(roidb, ratio_list, ratio_index,args.batch_size, imdb.num_classes, training=True)

    # 构建数据集迭代器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_batch,num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuad
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained = True, class_agnostic = args.class_agnostic)
    elif args.net == 'res18':
        fasterRCNN = resnet(imdb.classes, pretrained = True, class_agnostic = args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()
    fasterRCNN.creat_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        print("key:", key, value.requires_grad)
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr': lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value], 'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:  # 如果存在先前训练保存的模型，可加载模型继续训练
        load_name = os.path.join(output_dir,'faster_rcnn_{}_{}_{}.pth'.
                                 format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoing %s" % (load_name))

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0

        if (epoch + 1) % args.lr_decay_step == 0:  # 一定迭代次数之后调整学习率
            adjust_learning_rate(optimizer, args.le_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)   # data包含padding_data, im_info, gt_boxes_padding, num_boxes
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, \
            RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            # 多任务损失相加
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            if args.net == "vgg16":  # 这个是类似caffe里的提取梯度clip, 防止loss爆炸
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            # disp_interval 默认值为100，训练100幅图像后输出当前状态
            if step % args.disp_interval == 0:
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt, bg_cnt))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                loss_temp = 0

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
