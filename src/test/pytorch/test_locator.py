#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-18'

"""


import os
import unittest
from core import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from pytorch.locator import Locator

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestLocator(unittest.TestCase):
    def test_calcuate_location_features(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        loca = Locator(c)
        loca.calcuate_location_features([0.5], chosen=["Tumor_009"])

    def test_create_train_data(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        loca = Locator(c)
        features = loca.calcuate_location_features([0.5], chosen=None)
        X, Y = loca.create_train_data(features)

        save_filename = "{}/data/location_features.npz".format(c.PROJECT_ROOT)
        np.savez_compressed(save_filename, X=X, Y=Y)
        print(">>> >>> ", save_filename," saved!")


    def test_train_svm(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        load_filename = "{}/data/location_features.npz".format(c.PROJECT_ROOT)
        result = np.load(load_filename)
        X = result["X"]
        Y = result["Y"]

        loca = Locator(c)
        # features = loca.calcuate_location_features([0.5], chosen=None)
        # X, Y = loca.create_train_data(features)
        loca.train_svm(X, Y)

    def test_evaluate(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        load_filename = "{}/data/location_features.npz".format(c.PROJECT_ROOT)
        result = np.load(load_filename)
        X = result["X"]
        Y = result["Y"]

        loca = Locator(c)
        loca.evaluate(X, Y)

    def test_output_result_csv(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        loca = Locator(c)
        loca.output_result_csv([0.5], None)
