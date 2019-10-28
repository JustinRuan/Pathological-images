#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-18'

"""


import os
import unittest
from core import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from pytorch.locator import Locator
import csv
from pytorch.cancer_map import CancerMapBuilder

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestLocator(unittest.TestCase):

    def test_output_result_csv(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        loca = Locator(c)

        # select = ["Tumor_{:0>3d}".format(i) for i in range(1, 112)] # 112
        # loca.output_result_csv("csv_3", tag =64,  chosen=select)
        #
        # select = ["Normal_{:0>3d}".format(i) for i in range(1, 161)] #161
        # loca.output_result_csv("csv_3", tag=64, chosen=select)

        # select = ["Test_{:0>3d}".format(i) for i in range(1, 49)]   # 49
        # loca.output_result_csv("csv_4", tag=64, chosen=select)
        #
        # select = ["Test_{:0>3d}".format(i) for i in range(50, 131)] #131
        # loca.output_result_csv("csv_4", tag=64, chosen=select)

        # test
        code = [1, 2, 3, 5, 6, 7, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            , 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 47, 50, 51, 52, 54, 56, 57, 58, 59, 60
            , 61, 63, 64, 67, 68, 69, 70, 71, 72, 73, 75, 76, 78, 80, 81, 83, 85, 86, 87, 88, 90
            , 91, 92, 93, 94, 95, 96, 98, 100, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113
            , 115, 118, 119, 120, 121, 123, 124, 125, 126, 128, 129, 130]
        select = ["Test_{:0>3d}".format(i) for i in code] #131
        loca.output_result_csv("csv_6", tag=64, chosen=select)

    def test_csv2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        i = 1
        sub_path = "csv_1"
        code = "Tumor_{:0>3d}".format(i)
        filename = "{}/results/{}_history_v2.npz".format(c.PROJECT_ROOT, code)
        result = np.load(filename, allow_pickle=True)

        x1 = result["x1"]
        y1 = result["y1"]
        x2 = result["x2"]
        y2 = result["y2"]
        coordinate_scale = result["scale"]
        assert coordinate_scale == 1.25, "Scale is Error!"

        history = result["history"].item()

        cmb = CancerMapBuilder(c, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

        h, w = cancer_map.shape

        csv_filename = "{}/results/{}/{}.csv".format(c.PROJECT_ROOT, sub_path, code)
        x = []
        y = []
        p = []
        with open(csv_filename, 'r', )as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                p.append(float(item[0]))
                x.append(int(item[1]) // 32 - x1)
                y.append(int(item[2]) // 32 - y1)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/%s.tif" % code,
                                 'Train_Tumor/%s.xml' % code, code)

        src_img = np.array(imgCone.get_fullimage_byScale(1.25))
        mask_img = imgCone.create_mask_image(1.25, 0)
        mask_img = mask_img['C']

        roi_mask = mask_img[y1:y1 + h, x1:x1+w]
        src_img = src_img[y1:y1 + h, x1:x1+w, :]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        ax = axes.ravel()

        ax[0].imshow(mark_boundaries(src_img, roi_mask, color=(1, 0, 0), ))
        ax[0].scatter(x, y, c=p, cmap='Spectral')
        ax[0].set_title("image")

        ax[1].imshow(mark_boundaries(cancer_map, roi_mask, color=(1, 0, 0), ))
        im = ax[1].scatter(x, y, c=p, cmap='Spectral')
        ax[1].set_title("prob map")

        # fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()