#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-20'

"""

import unittest
import numpy as np
from core import Params, ImageCone, Open_Slide
from core.slic import SLICProcessor
from pytorch.segmentation import Segmentation
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation.slic_superpixels import slic

from EvalSPModule import computeASA, computeBR

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class TestSLIC(unittest.TestCase):

    def test_slic(self):
        f_map = np.load("feature_map.npy")
        print(f_map.shape)

        slic = SLICProcessor(f_map, 100, 0.4)
        label_map = slic.clusting(3, False)
        np.save("label_map", label_map)

    def test_show(self):

        test_set = {"001" : (2100, 3800, 2400, 4000),
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

        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * 1.25 / 1.25).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1
        block = imgCone.get_image_block(1.25, int(xx1 + (w >> 1)), int(yy1 + (h >> 1)), w, h)
        src_img = block.get_img()



        all_mask = imgCone.create_mask_image(1.25, 0)
        cancer_mask = all_mask['C']
        gt_img = cancer_mask[yy1:yy2, xx1:xx2]

        slic_map = slic(src_img, 100, 20, enforce_connectivity=True,)
        slic_img = label2rgb(slic_map, alpha=0.3, image=gt_img)

        label_map = np.load("label_map.npy")
        print(label_map.shape)
        label_img = label2rgb(label_map, alpha=0.3, image=gt_img)
        label_img2 = label2rgb(label_map, alpha=0.3, image=None)


        asa, error_list = computeASA(label_map.flatten().tolist(), gt_img.flatten().tolist(), 1)
        br = computeBR(label_map.flatten().tolist(), gt_img.flatten().tolist(), h, w, 1)
        print("feature slic", asa, br)

        asa, error_list = computeASA(slic_map.flatten().tolist(), gt_img.flatten().tolist(), 1)
        br = computeBR(slic_map.flatten().tolist(), gt_img.flatten().tolist(), h, w, 1)
        print("slic", asa, br)

        fig, axes = plt.subplots(2, 3, figsize=(100, 40), dpi=10)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(gt_img)
        ax[1].set_title("gt_img")

        ax[2].imshow(label_img)
        ax[2].set_title("label_img")

        ax[3].imshow(label_img2)
        ax[3].set_title("label_img2")

        ax[4].imshow(slic_img)
        ax[4].set_title("slic_img")

        for a in ax.ravel():
            a.axis('off')
        plt.show()


