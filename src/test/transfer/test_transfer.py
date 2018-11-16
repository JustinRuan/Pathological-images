#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

import unittest
from core import *
from transfer import Transfer

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin2.json"

class Test_transfer(unittest.TestCase):

    # def test_extract_features(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #     cnn = Transfer(c)
    #
    #     imgCone = ImageCone(c)
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
    #                              None, "17004930")
    #     seeds = [(12608, 17856), (23232, 22656), (7296, 14208)] # C, C, S
    #     result = cnn.extract_features(imgCone, 20, 256, seeds)
    #
    #     print(len(result), result[0].shape)

    # def test_fine_tuning(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #     cnn = Transfer(c)
    #
    #     # imgCone = ImageCone(c)
    #     #
    #     # # 读取数字全扫描切片图像
    #     # tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
    #     #                          '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")
    #
    #     cnn.fine_tuning_1("T_SC_2000_256")
    #
    # def test_predict(self):
    #     c = Params()
    #     c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
    #     cnn = Transfer(c)
    #
    #     imgCone = ImageCone(c)
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
    #                              None, "17004930")
    #     seeds = [(12608, 17856), (23232, 22656), (7296, 14208)]  # C, C, S
    #     result = cnn.predict(imgCone, 20, 256, seeds)
    #     print(result)

    def test_extract_features_for_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c)

        cnn.extract_features_for_train("inception_v3", "T_NC_2000_256", 100)

    def test_fine_tuning_data_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c)

        cnn.fine_tuning_top_model_saved_file("inception_v3", "T_NC_2000_256")

    def test_merge_save_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c)
        cnn.merge_save_model("inception_v3")

    def test_evaluate_entire_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c)

        cnn.evaluate_entire_model("inception_v3", "/trained/inception_v3-0040-0.07-0.99.ckpt",
                                  "T_NC_2000_256", 100)