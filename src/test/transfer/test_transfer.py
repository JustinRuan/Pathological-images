#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

import unittest
import numpy as np
from core import *
from transfer import Transfer

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

SAMPLE_FIlENAME = "T_NC_500_128"
# SAMPLE_FIlENAME = "T_NC_2000_256"
# SAMPLE_FIlENAME = "T_NC_4000_256"

# PATCH_TYPE = "4000_256"
# PATCH_TYPE = "2000_256"
PATCH_TYPE = "500_128"


# MODEL_NAME = "inception_v3"
MODEL_NAME = "densenet121"
# MODEL_NAME = "densenet169"
# MODEL_NAME = "densenet201"
# MODEL_NAME = "resnet50"
# MODEL_NAME = "inception_resnet_v2"
# MODEL_NAME = "vgg16"
# MODEL_NAME = "mobilenet_v2"
# MODEL_NAME = "nasnet"
# MODEL_NAME = ""

class Test_transfer(unittest.TestCase):

    # 检测特征提取的 稳定性
    def test_extract_features(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        # model = cnn.load_model(mode = 0, weights_file=c.PROJECT_ROOT+"/models/trained/inception_v3_500_128.ckpt")

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(8800, 12256)] * 10  # C, C, S
        result = cnn.extract_features(None, imgCone, 5, 128, seeds)
        print(np.std(result, axis = 1) )
        print(np.std(result, axis=0))
    #
    # def test_compare_weights(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #     cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
    #
    #     # model1 = cnn.load_model(mode=0, weights_file="G:/inception_v3_500_128/cp-0001-0.10-0.89.ckpt")
    #     model2 = cnn.load_model(mode=0, weights_file="G:/inception_v3_500_128/cp-0001-0.41-0.85.ckpt")
    #     full_mode = cnn.load_model(mode=0, weights_file=c.PROJECT_ROOT+ "/models/trained/inception_v3_500_128.ckpt")
    #     # base_mode = cnn.load_model(mode=0, weights_file="imagenet")
    #     for i, (f_layer, b_layer) in enumerate(zip(full_mode.layers, model2.layers)):
    #         fw = f_layer.get_weights()
    #         bw = b_layer.get_weights()
    #
    #         for f, t in zip(fw, bw):
    #             if len(f) > 0:
    #                 if (np.array(f) == np.array(t)).all():
    #                     print(i, "相等")
    #                 else:
    #                     print(i, "不相等")

        # print(full_mode.summary())
        # print(top_model.summary())
        # for i, (f_layer, b_layer) in enumerate(zip(full_mode.layers[312:], top_model.layers)):
        #     fw = f_layer.get_weights()
        #     bw = b_layer.get_weights()
        #     for f, t in zip(fw, bw):
        #         if len(f) > 0:
        #             if (np.array(f) == np.array(t)).all():
        #                 print(i, "相等")
        #             else:
        #                 print(i, "不相等")

    def test_extract_features_for_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        # cnn.extract_features_for_train("T_NC_4000_256", 100)
        # cnn.extract_features_for_train("T_NC_2000_256", 100)
        cnn.extract_features_for_train(SAMPLE_FIlENAME, 100)

    def test_fine_tuning_data_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)

        train_file = "{}_{}_train_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)
        test_file = "{}_{}_test_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)

        cnn.fine_tuning_top_cnn_model_saved_file(train_file, test_file,
                                                 batch_size=None, epochs=500, initial_epoch=0)


    def test_merge_save_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.merge_save_model()

    def test_evaluate_entire_cnn_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)

        cnn.evaluate_entire_cnn_model(SAMPLE_FIlENAME, 100)

    def test_evaluate_cnn_svm_rf_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)

        cnn.evaluate_cnn_svm_rf_model(SAMPLE_FIlENAME, 100)

    def test_predict(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        model = cnn.load_model(mode = 0)
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
        cnn.fine_tuning_model_with_freezed("T_NC_500_128",batch_size=32, freezed_num=311, epochs=500, initial_epoch=0)

    def test_train_top_svm(self):
        train_file = "{}_{}_train_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)
        test_file = "{}_{}_test_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)

        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.train_top_svm(train_file, test_file)

    def test_train_top_rf(self):
        train_file = "{}_{}_train_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)
        test_file = "{}_{}_test_features.npz".format(MODEL_NAME, SAMPLE_FIlENAME)

        c = Params()
        c.load_config_file(JSON_PATH)
        cnn = Transfer(c, MODEL_NAME, PATCH_TYPE)
        cnn.train_top_rf(train_file, test_file)