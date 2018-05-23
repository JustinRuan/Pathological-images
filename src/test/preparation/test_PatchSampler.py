#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import unittest
from core import *
from preparation import *
import matplotlib.pyplot as plt


class TestPatchSampler(unittest.TestCase):

    def test_load(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone.ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        self.assertTrue(tag)

        if tag:
            ps = PatchSampler.PatchSampler(c)
            highScale = c.EXTRACT_SCALE
            lowScale = c.GLOBAL_SCALE

            result = ps.generate_seeds4_high(imgCone, lowScale, highScale)
            print(result)

            ps.extract_patches_AZone(imgCone, highScale)

if __name__ == '__main__':
    unittest.main()
