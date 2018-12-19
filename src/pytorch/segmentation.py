#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""

import numpy as np
from pytorch.util import get_image_blocks_itor
from core.util import transform_coordinate
from pytorch.encoder import Encoder

class Segmentation(object):
    def __init__(self, params, src_image):
        self._params = params
        self._imgCone = src_image

    def get_seeds_for_seg(self, x1, y1, x2, y2, scale):
        x_set = np.arange(x1, x2)
        y_set = np.arange(y1, y2)
        xx, yy = np.meshgrid(x_set, y_set)
        results = []
        for x,y in zip(xx.flatten(), yy.flatten()):
            results.append((x ,y))
        return results

    def get_seeds_itor(self, seeds, seed_scale, extract_scale, patch_size, batch_size):
        extract_seeds = transform_coordinate(0,0, seed_scale, seed_scale, extract_scale, seeds)

        itor = get_image_blocks_itor(self._imgCone, extract_scale, extract_seeds, patch_size, patch_size, batch_size)
        return itor

    def create_feature_map(self, x1, y1, x2, y2, scale, extract_scale):
        patch_size = 32
        batch_size = 64

        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = \
            np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)

        global_seeds = self.get_seeds_for_seg(xx1, yy1, xx2, yy2, GLOBAL_SCALE)

        img_itor = self.get_seeds_itor(global_seeds, GLOBAL_SCALE, extract_scale, patch_size, batch_size)

        encoder = Encoder(self._params, "cae2", "cifar10")
        features = encoder.extract_feature(img_itor, len(global_seeds), batch_size)

        w = xx2 - xx1
        h = yy2 - yy1
        feature_map = np.zeros((h, w, 32))
        # feature_map的原点是全切片中检测区域的左上角（xx1，yy1），而提取特征时用的是全切片的坐标(0, 0)
        for (x, y), fe in zip(global_seeds, features):
            feature_map[y - yy1, x - xx1, :] = fe

        return feature_map
