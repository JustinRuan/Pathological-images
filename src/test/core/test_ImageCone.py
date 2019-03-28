#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-19'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
import numpy as np


class TestImageCone(unittest.TestCase):

    def test_load(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone(c, KFB_Slide(c.KFB_SDK_PATH))

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        self.assertTrue(tag)

        if tag:
            scale = c.GLOBAL_SCALE
            fullImage = imgCone.get_fullimage_byScale(scale)
            masks = imgCone.create_mask_image(scale,64)
            mask1 = imgCone.get_effective_zone(scale)
            mask2 = masks["S"] & mask1

            fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=200)
            ax = axes.ravel()

            ax[0].imshow(fullImage)
            ax[0].set_title("full")

            ax[1].imshow(masks["C"])
            ax[1].set_title("C_mask")
            ax[2].imshow(masks["S"])
            ax[2].set_title("S_mask")
            ax[3].imshow(masks["E"])
            ax[3].set_title("E_mask")
            ax[4].imshow(mask1)
            ax[4].set_title("ROI")
            ax[5].imshow(masks["L"])
            ax[5].set_title("L_mask")

            for a in ax.ravel():
                a.axis('off')

            plt.show()

    def test_load2(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json")
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_014.tif",
                                 'Train_Tumor/tumor_014.xml', "Tumor_014")
        self.assertTrue(tag)

        if tag:
            scale = c.GLOBAL_SCALE
            fullImage = imgCone.get_fullimage_byScale(scale)
            masks = imgCone.create_mask_image(scale,8)
            eff_region = imgCone.get_effective_zone(scale)

            fig, axes = plt.subplots(1, 6, figsize=(8, 3), dpi=200)
            ax = axes.ravel()

            ax[0].imshow(fullImage)
            ax[0].set_title("full")

            ax[1].imshow(masks["C"])
            ax[1].set_title("C_mask")
            ax[2].imshow(masks["N"])
            ax[2].set_title("S_mask")
            ax[3].imshow(masks["EI"])
            ax[3].set_title("EI_mask")
            # ax[4].imshow(masks["EO"])
            ax[4].imshow(np.bitwise_and(masks["EI"], masks["EO"]))
            ax[4].set_title("EO_mask")
            ax[5].imshow(eff_region)
            ax[5].set_title("eff_region")

            for a in ax.ravel():
                a.axis('off')

            plt.show()
