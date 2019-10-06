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

        # sc = SlideClassifier(c, "simple", "256")
        # sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=0.6)

        sc = SlideClassifier(c, "simple", "128")
        sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=1.0)

        sc = SlideClassifier(c, "simple", "64")
        sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=1.0)

        # sc = SlideClassifier(c, "simple", "32")
        # sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=0.6)

    def test_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "Slide_simple", "64")
        filename = "slide_train_data64_p1.0_np1.0.npz"
        sc.train(filename, class_weight=None, batch_size=200, loss_weight=0.001,epochs=100)

    def test_clear(self):
        path = "E:\Justin\WorkSpace\PatholImage\models\pytorch\Slide_simple_128"
        util.clean_checkpoint(path, best_number=10)

    def test_update_history(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "Slide_simple", "64")
        select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)] # range(1,112)
        sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "128")
        # select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)] # range(1,112)
        # sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "256")
        # select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)] # range(1,112)
        # sc.update_history(chosen=select, batch_size=100)

    def test_update_history2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideClassifier(c, "Slide_simple", "128")
        select = ["Test_{:0>3d}".format(i) for i in [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,
                                                      61,64,65,66,68,69,71,73,74,75,79,
                                                    82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]]
        sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "128")
        # sc.update_history(chosen=select, batch_size=100)
        #
        # sc = SlideClassifier(c, "Slide_simple", "256")
        # sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "64")
        # select = ["Test_{:0>3d}".format(i) for i in [27]]
        # sc.update_history(chosen=select, batch_size=100)