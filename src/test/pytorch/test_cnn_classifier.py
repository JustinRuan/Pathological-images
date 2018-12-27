#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import unittest
from core import Params, ImageCone, Open_Slide
from pytorch.cnn_classifier import CNN_Classifier
import torch

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "cifar10"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name=None, batch_size=32, epochs = 200)

    def test_train_model_patholImg(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "500_128"
        # sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name="T_NC_{}".format(sample_name), batch_size=32, epochs = 30)

    def test_predict_on_batch(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet_22"
        sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
        result = cnn.predict_on_batch(imgCone, 20, 256, seeds, 1)
        print(result)

    def test_Image_Dataset(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "500_128"
        # sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        train_data, test_data = cnn.load_custom_data("T_NC_{}".format(sample_name))
        print(train_data.__len__())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
        print(train_loader)

        for index, (x, y) in enumerate(train_loader):
            print(x.shape, y)
            if index > 10: break