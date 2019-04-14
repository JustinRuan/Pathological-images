#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-13'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
from skimage.io import imread
import random
import numpy as np

from preparation.augmentation import ImageAugmentation

# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"

class TestAugmentation(unittest.TestCase):

    def test_augment_images(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        # filename = "T_NC_Simple0330_4000_256_test.txt"
        filename = "T_NC_P0404_4000_256_test.txt"
        patch_path = c.PATCHS_ROOT_PATH["P0404"]
        all_file_list, _ = util.read_csv_file(patch_path, "{}/{}".format(patch_path, filename))

        N = 20
        rnd = random.randint(0, len(all_file_list) // N)
        file_list = all_file_list[rnd:rnd + N]

        augment = ImageAugmentation(l_range = (0.9, 1.1), a_range = (0.95, 1.05),
                                   b_range = (0.95, 1.05), constant_range = (-10, 10))

        fig = plt.figure(figsize=(16, 10), dpi=100)
        for index, filename in enumerate(file_list):
            img = imread(filename)
            result = augment.augment_images(img)
            plt.subplot(5, 2 * N / 5, 2 * index + 1)
            plt.imshow(img)

            plt.axis("off")
            plt.subplot(5, 2 * N / 5, 2 * index + 2)
            plt.axis("off")
            plt.imshow(result)

        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        plt.show()

    def test_augment_dataset(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        samples_name = ("P0327", "T_NC_Simple0327_2_4000_256_train.txt")

        augment = ImageAugmentation(l_range=(0.9, 1.1), a_range=(0.95, 1.05),
                                    b_range=(0.95, 1.05), constant_range=(-10, 10))

        augment.augment_dataset(c, samples_name, "Aug_LAB", range=(0, 100))
