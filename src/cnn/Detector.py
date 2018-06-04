#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""

from core import *
import caffe
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io, util
from sklearn.svm import NuSVC, SVC
from sklearn.externals import joblib
from skimage.draw import rectangle # 需要skimage 0.14及以上版本
from core.util import get_seeds
from cnn import transfer_cnn

class Detector(object):

    def __init__(self, config_file, slice_file, ano_filename, id_string):
        self._params = Params.Params()
        self._params.load_config_file(config_file)

        self.imgCone = ImageCone.ImageCone(self._params)

        # 读取数字全扫描切片图像
        tag = self.imgCone.open_slide(slice_file, ano_filename, id_string)

        if tag:
            w, h = self.imgCone.get_image_width_height_byScale(self._params.GLOBAL_SCALE)
            self.ImageWidth = w
            self.ImageHeight = h
            self.ROI = np.zeros((h, w), dtype=np.bool)
        return

    def get_ROI_img(self, x1, y1, x2, y2, coordinate_scale, img_scale):
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * img_scale / coordinate_scale).astype(np.int)
        block = self.imgCone.get_image_block(img_scale, xx1, yy1, xx2 - xx1, yy2 - yy1)
        return block.get_img()

    def set_ROI(self, x1, y1, x2, y2, scale):
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2 ])* GLOBAL_SCALE / scale).astype(np.int)
        rr, cc = rectangle((yy1, xx1), end=(yy2, xx2))
        self.ROI[rr, cc] = 1
        self.ROI_width = xx2 - xx1
        self.ROI_height = yy2 - yy1
        return

    def reset_ROI(self):
        self.ROI =  np.zeros((self.ImageHeight, self.ImageWidth), dtype=np.bool)
        return

    def get_points_ROI(self, highScale, patch_size_high, spacingHigh):
        return get_seeds(self.ROI, self._params.GLOBAL_SCALE, highScale,patch_size_high, spacingHigh, margin=0)

    # highScale, spacingHigh, block_size 都是在高分辨率下的值
    def get_imagelist_ROI(self, highScale, spacingHigh, block_size):
        seeds = self.get_points_ROI(highScale, block_size, spacingHigh)

        images = []
        for (x, y) in seeds:
            block = self.imgCone.get_image_block(highScale, x, y, block_size, block_size)
            images.append(block.get_img())
        return seeds, images

    def detect_ROI(self, scale, block_size):
        spacing = block_size / 2

        seeds, images = self.get_imagelist_ROI(scale, spacing, block_size)

        tc = transfer_cnn.transfer_cnn(self._params, "googlenet", "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt")
        tc.start_caffe()
        glcm_features, cnn_features = tc.extract_cnn_feature(images)
        # print(np.array(cnn_features)[:, 0:3])

        features = tc.comobine_features(glcm_features, cnn_features)
        classifier = tc.load_svm_model()
        predicted_tags = classifier.predict(features)
        return seeds, predicted_tags

    def draw_result(self, seeds, seeds_scale, block_size_high, tags, x1, y1, xy_scale):
        result = 255 * np.ones((self.ROI_width, self.ROI_height), dtype=np.int)
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        # 所有坐标都统一到 GLOBAL_SCALE 下
        xx1 = int(x1 * GLOBAL_SCALE / xy_scale)
        yy1 = int(y1 * GLOBAL_SCALE / xy_scale)
        block_size_low = int(block_size_high * GLOBAL_SCALE / seeds_scale)

        for ((x, y), tag) in zip(seeds, tags):
            xx = int(x * GLOBAL_SCALE / seeds_scale) - xx1
            yy = int(y * GLOBAL_SCALE / seeds_scale) - yy1
            print(xx, yy)
            rr, cc = rectangle((yy, xx), extent=(block_size_low, block_size_low))
            rr[rr >= self.ROI_height] = self.ROI_height -1
            cc[cc >= self.ROI_width] = self.ROI_width - 1
            result[rr, cc] = tag * 128

        return result