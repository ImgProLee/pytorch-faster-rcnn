#vgg16网络构建，加载预训练backbone网络

import torch
import torch.nn as nn
import torchvision.models as models
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN

class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        # 要继承父类的属性，需要在子类中调用父类的构造函数
        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)  #torch.load返回的是一个OrderedDict.
            # 加载模型时一般用 model.load_state_dict(torch.load(model_path))
            # Sequential定义的网络的模型参数不能与属性一层层”定义的网络直接对应
            # 这里k表示层，如features.0.weight、features.0.bias，v表示tensor参数
            # .state_dict()表示模型的参数，.load_state_dict()表示加载模型参数。
            vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        # not using the last maxpool layer，舍弃最后一个池化层，将前面的层赋给RCNN_base
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        #舍弃掉最后一个分类层的全连接层
        self.RCNN_top = vgg.classifier

        # not using the last classification layer，给出单独的分类层
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        #边框回归层
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
    def head_to_tail(self, pool5):
        # 将最后一个池化层展平
        pool5_flat = pool5.view(pool5.size(0), -1)
        #将展平的pool5输入到全连接层中
        fc7 = self.RCNN_top(pool5_flat)
        return fc7

