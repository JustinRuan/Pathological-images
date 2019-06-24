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
import csv

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

    def test_csv(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        i = 9
        code = "Tumor_{:0>3d}".format(i)
        # filename = "{}/results/{}_cancermap.npz".format(c.PROJECT_ROOT, code)
        # result = np.load(filename)
        # x1 = result["x1"]
        # y1 = result["y1"]
        # coordinate_scale = result["scale"]
        # cancer_map = result["cancer_map"]

        csv_filename = "{}/results/{}.csv".format(c.PROJECT_ROOT, code)
        x = []
        y = []
        p = []
        with open(csv_filename, 'r', )as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                p.append(float(item[0]))
                x.append(int(item[1]) // 32)
                y.append(int(item[2]) // 32)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/%s.tif" % code,
                                 'Train_Tumor/%s.xml' % code, code)

        src_img = imgCone.get_fullimage_byScale(1.25)
        mask_img = imgCone.create_mask_image(1.25, 0)
        mask_img = mask_img['C']

        plt.figure()
        plt.imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
        plt.scatter(x, y, c=p, cmap='Spectral')
        plt.colorbar()
        plt.show()
