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
from pytorch.encoder_factory import EncoderFactory
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from core.util import transform_coordinate

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

        # encoder = EncoderFactory(self._params, "idec", "AE_500_32", 16)
        encoder = EncoderFactory(c, "cae", "AE_500_32", 16)
        seg = Segmentation(c, imgCone, encoder)
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

    def test_create_superpixels(self):
        test_set = [("001", 2100, 3800, 2400, 4000),
                    ("003", 2400, 4700, 2600, 4850)]
        id = 1
        roi = test_set[id]
        slice_id = roi[0]
        x1 = roi[1]
        y1 = roi[2]
        x2 = roi[3]
        y2 = roi[4]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
                                 'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        seg = Segmentation(c, imgCone)
        f_map = seg.create_feature_map(x1, y1, x2, y2, 1.25, 5)
        label_map = seg.create_superpixels(f_map, 0.4, iter_num = 3)

        print(label_map.shape)
        np.save("label_map", label_map)


    def test_4(self):
        a = np.random.rand(8,8)
        print(a)

        b = a[::2, ::2]
        print(b)

    def test_create_superpixels_slic(self):
        test_set = [("001", 2100, 3800, 2400, 4000),
                    ("003", 2400, 4700, 2600, 4850)]
        id = 1
        roi = test_set[id]
        slice_id = roi[0]
        x1 = roi[1]
        y1 = roi[2]
        x2 = roi[3]
        y2 = roi[4]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_%s.tif" % slice_id,
                                 'Train_Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * 1.25 / 1.25).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1
        block = imgCone.get_image_block(1.25, int(xx1 + (w >> 1)), int(yy1 + (h >> 1)), w, h)
        src_img = block.get_img()

        all_mask = imgCone.create_mask_image(1.25, 0)
        cancer_mask = all_mask['C']
        gt_img = cancer_mask[yy1:yy2, xx1:xx2]

        seg = Segmentation(c, imgCone)
        label_map = seg.create_superpixels_slic(x1, y1, x2, y2, 1.25, 1.25, 30, 20)
        # slic_img = label2rgb(label_map, alpha=0.3, image=gt_img)

        # print(label_map.shape)
        # np.save("label_map_slic", label_map)

        seeds = seg.get_seeds_at_boundaries(label_map, x1, y1,1.25, 16)
        # draw_seeds = np.array(seeds)
        draw_seeds = np.array(transform_coordinate(x1, y1, 1.25, 1.25, 1.25, seeds))
        print(len(seeds), "\n", seeds)

        fig, axes = plt.subplots(1, 2, figsize=(30, 12), dpi=80)
        ax = axes.ravel()

        ax[0].imshow(mark_boundaries(src_img, gt_img, color=(1, 0, 0),))
        ax[0].set_title("src & gt_img")

        ax[1].imshow(mark_boundaries(src_img, label_map, color=(0, 1, 0),mode='thick'))
        ax[1].set_title("slic label_img")
        ax[1].scatter(draw_seeds[:,0], draw_seeds[:,1], c='r')

        for a in ax.ravel():
            a.axis('off')
        plt.show()