#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""

import unittest
import numpy as np
from core import Params, ImageCone, Open_Slide
from pytorch.segmentation import Segmentation

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class TestSegmentation(unittest.TestCase):

    def test_create_feature_map(self):
        test_set = {"001": (2100, 3800, 2400, 4000),
                    "003": (2400, 4700, 2600, 4850)}
        id = "003"
        roi = test_set[id]
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % id,
                                 'Tumor/tumor_%s.xml' % id, "Tumor_%s" % id)

        seg = Segmentation(c, imgCone)
        # global_seeds =  seg.get_seeds_for_seg(x1, y1, x2, y2, 1.25)
        # print(global_seeds)
        f_map = seg.create_feature_map(x1, y1, x2, y2, 1.25, 5)
        print(f_map.shape)
        np.save("feature_map", f_map)

    def test_2(self):
        w = 3
        h = 4
        feature_map = np.zeros((h, w, 2))
        seeds = [(0,0), (1,1), (2,2)]
        features = [[1,2], [3,4], [5,6]]
        for (x, y), fe in zip(seeds, features):
            feature_map[x, y, :] = fe

        print(feature_map.shape)

    def test_3(self):
        a = np.array([1,2,3])
        b = np.array([2,3,4])
        d = np.sum(np.power(a - b, 2))
        print(d)