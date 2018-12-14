#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import unittest
from core import Params
from pytorch.cnn_classifier import CNN_Classifier

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "cifar10"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name=None, batch_size=32, epochs = 30)

    def test_train_model_patholImg(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        # model_name = "densenet_22"
        sample_name = "500_128"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name="T_NC_{}".format(sample_name), batch_size=32, epochs = 30)