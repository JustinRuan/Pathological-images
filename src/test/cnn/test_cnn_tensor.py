#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-25'

"""


import unittest
from core import *
from cnn import cnn_tensor

class Test_cnn_tensor(unittest.TestCase):

    def test_training(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_tensor(c, "simplenet128", "CNN_R_500_128")
        cnn.training()

    def test_predict(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        cnn = cnn_tensor(c, "simplenet128")

        imgCone = ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
        seeds = [(8384, 4448), (8112, 4704)]
        result = cnn.predict(imgCone, 5, 128, seeds)

        print(result)
