#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

import unittest
from core import *
from transfer import Transfer

class Test_transfer(unittest.TestCase):

    def test_extract_features(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        cnn = Transfer(c)

        imgCone = ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 None, "17004930")
        seeds = [(12608, 17856), (23232, 22656), (7296, 14208)] # C, C, S
        result = cnn.extract_features(imgCone, 20, 256, seeds)

        print(len(result), result[0].shape)

    def test_fine_tuning(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        cnn = Transfer(c)

        # imgCone = ImageCone(c)
        #
        # # 读取数字全扫描切片图像
        # tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
        #                          '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")

        cnn.fine_tuning_1("T_SC_2000_256")