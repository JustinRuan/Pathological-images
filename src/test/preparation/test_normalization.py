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

        plt.figure(figsize=(10,6), dpi=100)
        for index, filename in enumerate(file_list):
            img = imread("{}/{}".format(patch_path, filename))
            result = ImageNormalization.normalize(img)

            plt.subplot(2, 5, index + 1)
            plt.axis("off")
            plt.imshow(result)

        plt.show()
