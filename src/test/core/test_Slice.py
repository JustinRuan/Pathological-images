#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-18'

"""

import unittest
from core import *
import matplotlib.pyplot as plt


class TestSlice(unittest.TestCase):

    def test_load_annotation(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/config/justin.json")
        slice = Slice.Slice(c.KFB_SDK_PATH)

        # 读取数字全扫描切片图像
        tag = slice.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")
        self.assertTrue(tag)

        if tag:
            ImageWidth, ImageHeight = slice.get_image_width_height_byScale(c.GLOBAL_SCALE)
            fullImage = slice.get_image_block(c.GLOBAL_SCALE, 0, 0, ImageWidth, ImageHeight)
            print(ImageWidth, ImageHeight)

            slice.read_annotation('D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')
            self.assertEqual(len(slice.ano_TUMOR_R), 1)
            self.assertEqual(len(slice.ano_TUMOR_A), 5)
            self.assertEqual(len(slice.ano_NORMAL_R), 1)
            self.assertEqual(len(slice.ano_NORMAL_A), 5)

            fig, axes = plt.subplots(2, 2, figsize=(4, 3), dpi=300)
            ax = axes.ravel()

            ax[0].imshow(fullImage)
            ax[0].set_title("img")

            for a in ax.ravel():
                a.axis('off')

            plt.show()


if __name__ == '__main__':
    unittest.main()
