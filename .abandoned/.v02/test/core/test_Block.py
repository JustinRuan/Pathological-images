#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-20'

"""

import unittest
from core import *
import matplotlib.pyplot as plt


class TestBlock(unittest.TestCase):

    def test_Block(self):
        b = Block.Block("17004930", 13824, 17600, 20, 0, 256, 256)
        print(b.encoding())

    def test_getBlock(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone.ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        self.assertTrue(tag)

        block = imgCone.get_image_block(20, 12000, 20000, 256, 256)
        img = block.get_img()

        plt.imshow(img)
        plt.show()

        block.save_img(c.PATCHS_ROOT_PATH)


    def test_loadBlock(self):
        filename = "D:/Study/breast/Patches/P0523/17004930_012000_020000_2000_0.jpg"
        block = Block.Block()
        block.load_img(filename)
        print(block.encoding())

        img = block.get_img()
        plt.imshow(img)
        plt.show()

        return

if __name__ == '__main__':
    unittest.main()
