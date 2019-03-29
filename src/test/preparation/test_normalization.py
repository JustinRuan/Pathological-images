#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
from skimage.io import imread
from preparation.normalization import ImageNormalization
JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"

class TestNormalization(unittest.TestCase):

    def test_normalization(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        file_list = ["S2000_256_cancer/17004930_030656_015872_2000_0.jpg",
                     "S2000_256_cancer/17004930_030656_015488_2000_0.jpg",
                     "S2000_256_cancer/17004930_018048_016128_2000_0.jpg",
                     "S2000_256_cancer/17004930_021824_023808_2000_0.jpg",
                     "S2000_256_cancer/17004930_014528_014080_2000_0.jpg",
                     "S2000_256_stroma/17004930_039552_032352_2000_0.jpg",
                     "S2000_256_stroma/17004930_039264_032256_2000_0.jpg",
                     "S2000_256_stroma/17004930_039072_031200_2000_0.jpg",
                     "S2000_256_stroma/17004930_038592_030144_2000_0.jpg",
                     "S2000_256_stroma/17004930_038784_029568_2000_0.jpg",]
        patch_path = c.PATCHS_ROOT_PATH

        fig = plt.figure(figsize=(8,10), dpi=100)
        for index, filename in enumerate(file_list):
            img = imread("{}/{}".format(patch_path, filename))
            # result = ImageNormalization.normalize(img, 64.4, 17.8, -14.9, 9.69, 4.87, 4.22) # 5 x 128
            # result = ImageNormalization.normalize(img, 62.8,19.4,-16.2,12.06,6.86,7.14)  # 20 x 256
            result = ImageNormalization.normalize_mean(img)
            plt.subplot(5, 4, 2 * index + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(5, 4, 2 * index + 2)
            plt.axis("off")
            plt.imshow(result)

        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        plt.show()

    def test_calculate_mean_std(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = ImageNormalization(c)
        avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b = \
            normal.calculate_avg_mean_std(["T_NC_x_256_test.txt"])

        print(avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b)
        # 75.8213814145 12.1614585447 -4.43519343475 14.7552670861 5.36667117015 3.29930331455
