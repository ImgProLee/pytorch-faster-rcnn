
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from lib import datasets
import numpy as np
from lib.model.utils.config import cfg
from lib.datasets.factory import get_imdb
import PIL
import pdb


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    cache_file = os.path.join(imdb.cache_path, imdb.name + '_sizes.pkl')
    if os.path.exists(cache_file):
        print('Image sizes loaded from %s' % cache_file)
        with open(cache_file, 'rb') as f:
            sizes = pickle.load(f)
    # 第一次运行需要读取并存储序列化文件pkl，文件中保存了所有图像的尺寸信息列表
    else:
        print('Extracting image sizes... (It may take long time)')
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]
        with open(cache_file, 'wb') as f:
            pickle.dump(sizes, f)
        print('Done!!!')
    # 输出测试
    print("first roidb", roidb[1]['gt_overlaps'].toarray())
    # {'boxes': array([[ 33,  10, 447, 292]], dtype=uint16),
    # 'gt_classes': array([20], dtype=int32),
    # 'gt_ishard': array([0], dtype=int32),
    # 'gt_overlaps': <1x21 sparse matrix of type '<class 'numpy.float32'>'with 1 stored elements in Compressed Sparse Row format>,
    # 'flipped': False, 'seg_areas': array([117445.], dtype=float32)}
    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        # 'image': '/home/lzm/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/JPEGImages/2008_000002.jpg',
        # width': 500, 'height': 375, 'max_classes': array([20]), 'max_overlaps': array([1.], dtype=float32)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        # gt_overlaps是一个box_num*classes_num的矩阵，应该是每个box在不同类别的得分
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        # 返回每一行（每一个box）的最大值
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        # 返回每一行最大值的索引
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def filter_roidb(roidb):
    # filter the image without bounding box.
    # 过滤掉不包含边框的图像
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def rank_roidb_ratio(roidb):
    # 该方法检查图像的宽高比，将宽高比大于2或者小于0.5的图像的裁剪标志设置为True，
    # 并将宽高比更新为最大值或最小值，对训练图像进行裁剪时需要。
    # 构建roibatchLoader时需要对图像及其边框进行裁剪
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.
    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)
        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0
        ratio_list.append(ratio)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)  # 将ratio从小到大排序，返回排序索引
    return ratio_list[ratio_index], ratio_index  # 返回排序后的ratio_list以及对应的图像索引

def combined_roidb(imdb_names, training=True):
    def get_training_roidb(imdb):
        if cfg.TRAIN.USE_FLIPPED:   # USE_FLIPPED = True
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()   # 水平翻转gt_boxes
            print('done')
        print('Prapareing training data...')
        prepare_roidb(imdb)   # roidb.prepare_roidb()
        # prepare_roidb对每张图片的roidb进行信息扩充，添加id，路径，宽高，box类别信息等
        #ratio_index = rank_roidb_ratio(imdb)
        print('done')
        return imdb.roidb
    def get_roidb(imdb_name):
        # 调用factory文件中的get_imdb()方法
        imdb = get_imdb(imdb_name)   # imdb为调用get_imdb返回的pascal voc类对象
        print('Loaded dataset {:s}'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)   # gt
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    # s = 'voc_2007_trainval' 调用get_roidb()
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        # 由于前面定义的imdb对象在get_roidb函数中，若返回该对象需再次定义
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)  # 过滤不含box的roidb

    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    # 返回从小到大排序的ratio_list以及对应的原图roidb像索引
    return imdb, roidb, ratio_list, ratio_index