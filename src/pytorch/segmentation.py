#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""

import numpy as np
from pytorch.util import get_image_blocks_itor
from core.util import transform_coordinate
from pytorch.encoder_factory import EncoderFactory
from core.slic import SLICProcessor

class Segmentation(object):
    def __init__(self, params, src_image):
        '''
        构造分割器
        :param params:系统参数
        :param src_image:切片文件
        '''
        self._params = params
        self._imgCone = src_image

    def get_seeds_for_seg(self, x1, y1, x2, y2):
        '''
        得到（x1, y1）到（x2, y2）矩形内的各个点的坐标
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :return: 矩形内的整数坐标（种子点）集合
        '''
        x_set = np.arange(x1, x2)
        y_set = np.arange(y1, y2)
        xx, yy = np.meshgrid(x_set, y_set)
        results = []
        for x,y in zip(xx.flatten(), yy.flatten()):
            results.append((x ,y))
        return results

    def get_seeds_itor(self, seeds, seed_scale, extract_scale, patch_size, batch_size):
        '''
        根据Seeds集合变换到指定倍镜下，生成以种子点为中点的图块迭代器
        :param seeds: 种子点集合（x，y）
        :param seed_scale: 种子点坐标所在倍镜数
        :param extract_scale: 提取图块所用的倍镜数
        :param patch_size: 图块的大小
        :param batch_size: 批量数
        :return: 返回图块的Pytorch的Tensor的迭代器
        '''
        extract_seeds = transform_coordinate(0,0, seed_scale, seed_scale, extract_scale, seeds)

        itor = get_image_blocks_itor(self._imgCone, extract_scale, extract_seeds, patch_size, patch_size, batch_size)
        return itor

    def create_feature_map(self, x1, y1, x2, y2, scale, extract_scale):
        '''
        生成（x1,y1）到（x2,y2)矩形范围内的特征 矩阵
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :param scale: 以上四个坐标所在倍镜数
        :param extract_scale: 提取图块的特征所用的倍镜数
        :return:特征 矩阵
        '''
        patch_size = 32
        batch_size = 64

        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = \
            np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)

        global_seeds = self.get_seeds_for_seg(xx1, yy1, xx2, yy2)

        img_itor = self.get_seeds_itor(global_seeds, GLOBAL_SCALE, extract_scale, patch_size, batch_size)

        encoder = EncoderFactory(self._params, "cae", "cifar10", 32)
        features = encoder.extract_feature(img_itor, len(global_seeds), batch_size)
        f_size = len(features[0])

        w = xx2 - xx1
        h = yy2 - yy1
        feature_map = np.zeros((h, w, f_size))
        # feature_map的原点是全切片中检测区域的左上角（xx1，yy1），而提取特征时用的是全切片的坐标(0, 0)
        for (x, y), fe in zip(global_seeds, features):
            feature_map[y - yy1, x - xx1, :] = fe

        return feature_map

    def create_superpixels(self, feature_map, M, iter_num = 10):
        '''
        根据特征矩阵进行超像素分割
        :param feature_map: 特征矩阵
        :param M: 分割算法的权重系数
        :param iter_num: 分割算法所运行的迭代次数
        :return: label标记矩阵
        '''
        h, w, _ = feature_map.shape
        K = h * w // 32

        slic = SLICProcessor(feature_map, K, M)
        label_map = slic.clusting(iter_num = iter_num, enforce_connectivity = True,
                                  min_size_factor=0.1, max_size_factor=3.0)

        return label_map


