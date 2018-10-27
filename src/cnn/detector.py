#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-26'

"""
from core import *
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io, util
from sklearn.svm import SVC
from sklearn.externals import joblib
from skimage import segmentation
from skimage.draw import rectangle # 需要skimage 0.14及以上版本
from core.util import get_seeds
import random
from cnn import cnn_tensor

class Detector(object):

    def __init__(self, params, src_image):
        self._params = params
        self._imgCone = src_image

        w, h = self._imgCone.get_image_width_height_byScale(self._params.GLOBAL_SCALE)
        self.ImageWidth = w
        self.ImageHeight = h
        self.valid_map = np.zeros((h, w), dtype=np.bool)

        return

    def setting_detected_area(self, x1, y1, x2, y2, scale):
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2 ])* GLOBAL_SCALE / scale).astype(np.int)
        rr, cc = rectangle((yy1, xx1), end=(yy2, xx2))
        self.valid_map[rr, cc] = 1
        self.valid_area_width = xx2 - xx1
        self.valid_area_height = yy2 - yy1
        return

    def reset_detected_area(self):
        '''
        清空ROI中的标记
        :return:
        '''
        self.valid_map =  np.zeros((self.ImageHeight, self.ImageWidth), dtype=np.bool)
        return

    def get_points_detected_area(self, extract_scale, patch_size_extract, spacing_extract):
        '''
        得到ROI区中的点在高分辨率下的坐标
        :param extract_scale: 提取图时时，所使用的高分辨率对应的倍镜数
        :param patch_size_extract: 高分辨率下的图块大小
        :param spacing_extract: 高分辨率下的图块之间的距离
        :return: （x，y）种子点的集合
        '''
        return get_seeds(self.valid_map, self._params.GLOBAL_SCALE, extract_scale,patch_size_extract, spacing_extract, margin=8)

    def detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size):
        self.setting_detected_area(x1, y1, x2, y2, coordinate_scale)
        seeds = self.get_points_detected_area(extract_scale, patch_size, patch_size >> 1)

        cnn = cnn_tensor(self._params, "simplenet128")
        predictions = cnn.predict(self._imgCone, extract_scale, patch_size, seeds)

        # result = []
        # for (x, y), (class_id, probability) in zip(seeds, predictions):
        #     result.append((x, y, class_id, probability))

        return seeds, predictions

    def transform_coordinate(self, x1, y1, coordinate_scale, seeds_scale, target_scale, seeds):
        xx1 = (x1 * target_scale / coordinate_scale)
        yy1 = (y1 * target_scale / coordinate_scale)

        results = []
        for x, y in seeds:
            xx = int(x * target_scale / seeds_scale - xx1)
            yy = int(y * target_scale / seeds_scale - yy1)
            # xx = max(0, xx)
            # yy = max(0, yy)
            results.append((xx, yy))
        # print(results)
        return results

    def create_cancer_map(self, x1, y1, coordinate_scale, seeds_scale, target_scale, seeds,
                          predictions, seeds_patch_size):
        new_seeds = self.transform_coordinate(x1, y1, coordinate_scale, seeds_scale, target_scale, seeds)
        target_patch_size = int(seeds_patch_size * target_scale / seeds_scale)
        half = int(target_patch_size>>1)

        cancer_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)

        for (x, y), (class_id, probability)  in zip(new_seeds, predictions):
            if class_id == 1 :
                xx = x - half
                yy = y - half
                rr, cc = rectangle((yy, xx), extent=(target_patch_size, target_patch_size))

                select_y = (rr >= 0) & (rr < self.valid_area_height)
                select_x = (cc >= 0) & (cc < self.valid_area_width)
                select = select_x & select_y
                cancer_map[rr[select], cc[select]] = cancer_map[rr[select], cc[select]] + probability

        return cancer_map

    def get_detect_area_img(self, x1, y1, x2, y2, coordinate_scale, img_scale):

        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * img_scale / coordinate_scale).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1
        block = self._imgCone.get_image_block(img_scale, int(xx1 + (w >> 1)), int(yy1 + (h >> 1)), w, h)
        return block.get_img()