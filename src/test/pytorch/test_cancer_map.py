#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-08-02'

"""


import os
import unittest
from core import *
from pytorch.cancer_map import Slide_CNN, SlideFilter

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestCancerMapBuilder(unittest.TestCase):
    def test_save_train_data(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # sc = SlideClassifier(c, "simple", "256")
        # sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=0.6)

        sc = SlideFilter(c, "simple", "128")
        sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=1.0)

        sc = SlideFilter(c, "simple", "64")
        sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=1.0)

        # sc = SlideClassifier(c, "simple", "32")
        # sc.read_train_data("Train_Tumor", chosen=None, np_ratio=1.0, p_ratio=0.6)

    def test_train(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sc = SlideFilter(c, "Slide_simple", "64")
        filename = "slide_train_data64_p1.0_np1.0.npz"
        sc.train(filename, class_weight=None, batch_size=200, loss_weight=0.001,epochs=100)

    def test_clear(self):
        path = "E:\Justin\WorkSpace\PatholImage\models\pytorch\Slide_simple_128"
        util.clean_checkpoint(path, best_number=10)

    def test_update_history(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # tag = "Tumor"
        tag = "Test"
        # tag = "Test_Normal"
        # tag = "Normal"

        if tag == "Tumor":
            select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)]
            # select = ["Tumor_{:0>3d}".format(i) for i in range(1,20)]
        elif tag == "Test":

            # select = ["Test_{:0>3d}".format(i) for i in [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,
            #                                               61,64,65,66,68,69,71,73,74,75,79,
            #                                             82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]]
            # select = ["Test_{:0>3d}".format(i) for i in [4,10,29,30,33,38,48,66,79,84,99,102,116,117,122]]
            select = ["Test_{:0>3d}".format(i) for i in [117]]
        elif tag == "Test_Normal":
            temp = [3, 5, 6, 7, 9, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 28, 31, 32, 34, 35, 36, 37, 39, 41, 42, 43, 44, 45,
             47, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 67, 70, 72, 76, 77, 78, 80, 81, 83, 85, 86, 87, 88, 89, 91,
             92, 93, 95, 96, 98, 100, 101, 103, 106, 107, 109, 111, 112, 114, 115, 118, 119, 120, 123, 124, 125, 126, 127,
             128, 129, 130]
            select = ["Test_{:0>3d}".format(i) for i in temp]
        elif tag == "Normal":
            select = ["Normal_{:0>3d}".format(i) for i in range(1,161)]
            # select = ["Normal_{:0>3d}".format(i) for i in range(1, 65)]

        sc = SlideFilter(c, "Slide_simple", "64")
        sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "128")
        # sc.update_history(chosen=select, batch_size=100)

        # sc = SlideClassifier(c, "Slide_simple", "256")
        # sc.update_history(chosen=select, batch_size=100)
