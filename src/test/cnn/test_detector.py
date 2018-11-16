#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""


import unittest
from core import *
import matplotlib.pyplot as plt
from cnn import *
import numpy as np
from transfer import Transfer

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"

class Test_detector(unittest.TestCase):

    def test_detect2(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 'Tumor/tumor_004.xml', "Tumor_004")

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        x1 = 2000
        y1 = 2900
        x2 = 2500
        y2 = 3200
        seeds, predictions = detector.detect_region(x1, y1, x2, y2, 1.25, 5, 128)
        new_seeds, predictions_deep = detector.detect_region_detailed(seeds, predictions, 5, 128, 20, 256)
        # print(predictions_deep)

        cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 5, 1.25, seeds, predictions, 128, None,
                                                                     None)
        cancer_map2, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 20, 1.25, new_seeds, predictions_deep,
                                                                      256, prob_map, count_map)

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        np.savez("{}/data/cancer_map".format(c.PROJECT_ROOT), src_img, cancer_map, cancer_map2)

        fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(src_img)
        ax[1].imshow(cancer_map, alpha=0.6)
        ax[1].set_title("cancer_map")

        ax[2].imshow(src_img)
        ax[2].imshow(cancer_map2, alpha=0.6)
        ax[2].set_title("cancer_map2")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return


    def test_show_result(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        D = np.load("{}/data/cancer_map.npz".format(c.PROJECT_ROOT))
        src_img = D['arr_0']
        cancer_map = D['arr_1']
        cancer_map2 = D['arr_2']

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 'Tumor/tumor_004.xml', "Tumor_004")

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        x1 = 2000
        y1 = 2900
        x2 = 2500
        y2 = 3200

        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        detector.evaluate(0.5, cancer_map, mask_img)

        detector.evaluate(0.5, cancer_map2, mask_img)

        fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(src_img)
        ax[1].imshow(cancer_map, alpha=0.6)
        ax[1].set_title("cancer_map")

        ax[2].imshow(src_img)
        ax[2].imshow(cancer_map2, alpha=0.6)
        ax[2].set_title("cancer_map2")

        ax[3].imshow(src_img)
        ax[3].imshow(mask_img, alpha=0.6)
        ax[3].set_title("mask_img")

        for a in ax.ravel():
            a.axis('off')

        plt.show()


