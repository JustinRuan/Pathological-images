#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-08-02'

"""


import os
import unittest
from core import *
from pytorch.cancer_map import Slide_CNN, SlideClassifier

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"


class TestCancerMapBuilder(unittest.TestCase):
    def test_save_train_data(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "simple", "128")
        # scnn.save_train_data("Train_Tumor", chosen=["Tumor_001", ])
        sc.read_train_data("Train_Tumor", chosen=None)

    def test_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "simple", "128")
        sc.train(batch_size=100, loss_weight=0.001,epochs=10)

    def test_update_history(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "simple", "128")
        sc.update_history(chosen=None)

