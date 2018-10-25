#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-19'

"""

import unittest
from core import *
import matplotlib.pyplot as plt


class TestImageCone(unittest.TestCase):

    def test_load(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        self.assertTrue(tag)

        if tag:
            scale = c.GLOBAL_SCALE
            fullImage = imgCone.get_fullimage_byScale(scale)
            C_mask, N_mask, E_mask, L_mask = imgCone.create_mask_image(scale,64)
            mask1 = imgCone.get_effective_zone(scale)
            mask2 = N_mask & mask1

            fig, axes = plt.subplots(2, 3, figsize=(6, 3), dpi=400)
            ax = axes.ravel()

            ax[0].imshow(fullImage)
            ax[0].set_title("full")

            ax[1].imshow(C_mask)
            ax[1].set_title("C_mask")
            ax[2].imshow(N_mask)
            ax[2].set_title("N_mask")
            ax[3].imshow(E_mask)
            ax[3].set_title("E_mask")
            ax[4].imshow(mask1)
            ax[4].set_title("ROI")
            ax[5].imshow(L_mask)
            ax[5].set_title("L_mask")

            for a in ax.ravel():
                a.axis('off')

            plt.show()


if __name__ == '__main__':
    unittest.main()
