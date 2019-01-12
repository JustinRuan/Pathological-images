#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-07'

"""

import unittest
from core import *
from cnn.cnn_classifier import CNN_Classifier

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
JSON_PATH = "D:\code\python\Pathological-images\config\myConfig.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_40"
        sample_name = "cifar10"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model_cifar(batch_size=32, epochs = 100, initial_epoch = 0)

    def test_training(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet_22"
        # sample_name = "500_128"
        sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model("T_NC_{}".format(sample_name), batch_size=16, augmentation=(False, False), epochs=500, initial_epoch=0)

    def test_predict(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet_22"
        sample_name = "500_128"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(8768, 12192), (9152, 12128), (3328, 12032)] # C, C, S
        result = cnn.predict(imgCone, 5, 128, seeds, "simplenet128/cp-0031-0.26-0.89.ckpt")
        print(result)

        result = cnn.predict_on_batch(imgCone, 5, 128, seeds, 20, "simplenet128/cp-0031-0.26-0.89.ckpt")
        print(result)

    def test_predict_test_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        model_name = "densenet_22"
        sample_name = "500_128"

        cnn = CNN_Classifier(c, model_name, sample_name)

        cnn.predict_test_file(cnn.model_root + "/cp-0034-0.2121-0.9372.h5",
                              ["T_NC_500_128_test.txt"], 16)