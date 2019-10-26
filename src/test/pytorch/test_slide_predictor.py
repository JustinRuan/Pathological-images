#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-10-20'

"""

import unittest
from core import *
from pytorch.slide_predictor import SlidePredictor

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestSlidePredictor(unittest.TestCase):
    def test_extract_feature(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)

        Tumor_names = ["Tumor_{:0>3d}".format(i) for i in range(1, 112)] # 1, 112
        Normal_names = ["Normal_{:0>3d}".format(i) for i in range(1, 161)] # 1, 161

        feature_data, label_data = sp.extract_slide_features(tag=0, normal_names=Normal_names, tumor_names=Tumor_names)
        sp.save_train_data(feature_data, label_data, filename="slide_predictor_data_h5.npz", append=False)

    def test_train_svm(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)
        sp.train_svm(file_name="slide_predictor_data_h5.npz", test_name="slide_predictor_testdata_h5.npz")

    def test_extract_testdata_feature(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)
        Noraml = [3, 5, 6, 7, 9, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 28, 31, 32, 34, 35, 36, 37, 39, 41, 42, 43,
                  44, 45, 47, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 67, 70, 72, 76, 77, 78, 80, 81, 83, 85,
                  86, 87, 88, 89, 91, 92, 93, 95, 96, 98, 100, 101, 103, 106, 107, 109, 111, 112, 114, 115, 118, 119,
                  120, 123, 124, 125, 126, 127, 128, 129, 130]
        Tumor = [1, 2, 4, 8, 10, 11, 13, 16, 21, 26, 27, 29, 30, 33, 38, 40, 46, 48, 51, 52,
                 61, 64, 65, 66, 68, 69, 71, 73, 74, 75, 79,
                 82, 84, 90, 94, 97, 99, 102, 104, 105, 108, 110, 113, 116, 117, 121, 122]
        Tumor_names = ["Test_{:0>3d}".format(i) for i in Tumor]
        Normal_names = ["Test_{:0>3d}".format(i) for i in Noraml]

        feature_data, label_data = sp.extract_slide_features(tag=0, normal_names=Normal_names, tumor_names=Tumor_names)
        sp.save_train_data(feature_data, label_data, filename="slide_predictor_testdata_h5.npz", append=False)


    def test_train_test(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)
        sp.train_test(file_name="slide_predictor_data_h5.npz", test_name="slide_predictor_testdata_h5.npz")

    def test_train_s3vm(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)
        for i in range(1):
            sp.train_s3vm(file_name="slide_predictor_data_h5.npz", test_name="slide_predictor_testdata_h5.npz")

    def test_data_augment(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        sp = SlidePredictor(c)
        sp.data_augment("slide_predictor_data_d5.npz", "slide_predictor_augdata_d5.npz", count = 500)