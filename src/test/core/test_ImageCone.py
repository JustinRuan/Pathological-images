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
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/config/justin.json")
        imgCone = ImageCone.ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        self.assertTrue(tag)

        if tag:
            scale = c.GLOBAL_SCALE
            fullImage = imgCone.get_fullimage_byScale(scale)
            mask1 = imgCone.create_mask_image(scale, "TA")
            mask2 = imgCone.create_mask_image(scale, "TR")
            mask3 = imgCone.create_mask_image(scale, "NA")
            mask4 = imgCone.create_mask_image(scale, "NR")

            fig, axes = plt.subplots(2, 3, figsize=(4, 3), dpi=300)
            ax = axes.ravel()

            ax[0].imshow(fullImage)
            ax[0].set_title("full")

            ax[1].imshow(mask1)
            ax[1].set_title("TA")
            ax[2].imshow(mask2)
            ax[2].set_title("TR")
            ax[3].imshow(mask3)
            ax[3].set_title("NA")
            ax[4].imshow(mask4)
            ax[4].set_title("NR")

            for a in ax.ravel():
                a.axis('off')

            plt.show()


if __name__ == '__main__':
    unittest.main()
