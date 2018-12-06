#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-07'

"""

import unittest
from core import *
from cnn.cnn_classifier import CNN_Classifier

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        cnn = CNN_Classifier(c, "simple_cnn", "cifar10")
        cnn.train_model_cifar(batch_size=32, epochs = 20, initial_epoch = 0)