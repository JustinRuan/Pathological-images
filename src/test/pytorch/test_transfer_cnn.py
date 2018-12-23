#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-14'

"""

import unittest
from core import Params
from pytorch.transfer_cnn import Transfer
from core import Params, ImageCone, Open_Slide

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

SAMPLE_FIlENAME = "T_NC_500_128"
# SAMPLE_FIlENAME = "T_NC_2000_256"
# SAMPLE_FIlENAME = "T_NC_4000_256"

# PATCH_TYPE = "4000_256"
# PATCH_TYPE = "2000_256"
PATCH_TYPE = "500_128"


# MODEL_NAME = "inception_v3"
MODEL_NAME = "densenet121"

class Test_transfer_cnn(unittest.TestCase):

    def test_extract_features_save_to_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        fe = Transfer(c, MODEL_NAME, PATCH_TYPE)
        fe.extract_features_save_to_file(SAMPLE_FIlENAME, batch_size=32)

    def test_train_top_svm(self):
        MODEL_NAME = "densenet121"

        # SAMPLE_FIlENAME = "T_NC_500_128"
        # SAMPLE_FIlENAME = "T_NC_2000_256"
        SAMPLE_FIlENAME = "T_NC_4000_256"

        PATCH_TYPE = "4000_256"
        # PATCH_TYPE = "2000_256"
        # PATCH_TYPE = "500_128"

        train_file = "{}_{}_train_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)
        test_file = "{}_{}_test_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)

        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.train_top_svm(train_file, test_file)

    def test_save_pretrained_base_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        fe = Transfer(c, MODEL_NAME, None)
        fe.save_pretrained_base_model()

    def test_extract_feature(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet121"

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,

        fe = Transfer(c, model_name, None)
        # result = fe.extract_feature(imgCone, 20, 256, seeds, 2)
        result = fe.svm_predict_on_batch(imgCone, 20, 256, seeds, 2)
        print(result)


    def test_train_top_cnn(self):
        MODEL_NAME = "densenet121"

        # SAMPLE_FIlENAME = "T_NC_500_128"
        # SAMPLE_FIlENAME = "T_NC_2000_256"
        SAMPLE_FIlENAME = "T_NC_4000_256"

        PATCH_TYPE = "4000_256"
        # PATCH_TYPE = "2000_256"
        # PATCH_TYPE = "500_128"

        train_file = "{}_{}_train_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)
        test_file = "{}_{}_test_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)

        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.train_top_cnn_model(train_file, test_file, batch_size=100, epochs=30)

    def test_predict_on_batch(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet121"
        PATCH_TYPE = "2000_256"

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,

        cnn = Transfer(c, model_name, PATCH_TYPE)
        # result = fe.extract_feature(imgCone, 20, 256, seeds, 2)
        result = cnn.predict_on_batch(imgCone, 20, 256, seeds, 2)
        print(result)