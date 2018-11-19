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
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

# PATCH_TYPE = "4000_256"
# PATCH_TYPE = "2000_256"
PATCH_TYPE = "500_128"


MODEL_NAME = "inception_v3"

class Test_transfer(unittest.TestCase):

    def test_extract_features_for_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.extract_features_for_train("T_NC_4000_256", 100)
        # cnn.extract_features_for_train("T_NC_2000_256", 100)
        # cnn.extract_features_for_train("T_NC_500_128", 100)

    def test_fine_tuning_data_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)

        #cnn.fine_tuning_top_model_saved_file("T_NC_2000_256")
        # cnn.fine_tuning_top_model_saved_file("T_NC_500_128", batch_size=200, epochs=200, initial_epoch=100)
        cnn.fine_tuning_top_model_saved_file("T_NC_4000_256", batch_size=200, epochs=400, initial_epoch=300)

    def test_merge_save_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.merge_save_model()

    def test_evaluate_entire_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)

        # cnn.evaluate_entire_model("/trained/inception_v3_500_128-0219-0.33-0.90.ckpt",
        #                           "T_NC_500_128", 100)
        # cnn.evaluate_entire_model("/trained/inception_v3_2000_256-0286-0.20-0.95.ckpt",
        #                           "T_NC_2000_256", 100)
        cnn.evaluate_entire_model("/trained/inception_v3_4000_256-0396-0.33-0.88.ckpt",
                                  "T_NC_4000_256", 100)

    def test_predict(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        model = cnn.load_model(mode = 0, weights_file="/trained/inception_v3_2000_256-0286-0.20-0.95.ckpt")
        model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34880, 48960), (36224, 49920), (13312, 57856)]  # C, C, S
        result = cnn.predict(model, imgCone, 20, 256, seeds)
        print(result)

        result = cnn.predict_on_batch(model, imgCone, 20, 256, seeds, 2)
        print(result)

    def test_fine_tuning_inception_v3_249(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.fine_tuning_inception_v3_249("T_NC_500_128",100, 3, 0)