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

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestLocator(unittest.TestCase):
    def test_calcuate_location_features(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        loca = Locator(c)
        loca.calcuate_location_features([0.5], chosen=["Tumor_001"])

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
        # loca.output_result_csv("csv", chosen=["Tumor_009", "Tumor_011", "Tumor_016", "Tumor_026", "Tumor_039",
        #                                    "Tumor_047", "Tumor_058", "Tumor_068","Tumor_072","Tumor_076"])
        # loca.output_result_csv("csv_2", chosen=None)
        # loca.output_result_csv("csv_2", chosen=None)
        loca.output_result_csv("csv_2", chosen=["Tumor_076"])

    def test_csv2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        i = 76
        sub_path = "csv"
        code = "Tumor_{:0>3d}".format(i)
        filename = "{}/results/{}_cancermap.npz".format(c.PROJECT_ROOT, code)
        result = np.load(filename)
        x1 = result["x1"]
        y1 = result["y1"]
        coordinate_scale = result["scale"]
        cancer_map = result["cancer_map"]
        h, w = cancer_map.shape

        csv_filename = "{}/results/{}/{}.csv".format(c.PROJECT_ROOT, sub_path, code)
        x = []
        y = []
        p = []
        with open(csv_filename, 'r', )as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                p.append(float(item[0]))
                x.append(int(item[1]) // 32 - x1)
                y.append(int(item[2]) // 32 - y1)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/%s.tif" % code,
                                 'Train_Tumor/%s.xml' % code, code)

        src_img = np.array(imgCone.get_fullimage_byScale(1.25))
        mask_img = imgCone.create_mask_image(1.25, 0)
        mask_img = mask_img['C']

        roi_mask = mask_img[y1:y1 + h, x1:x1+w]
        src_img = src_img[y1:y1 + h, x1:x1+w, :]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        ax = axes.ravel()

        ax[0].imshow(mark_boundaries(src_img, roi_mask, color=(1, 0, 0), ))
        ax[0].scatter(x, y, c=p, cmap='Spectral')
        ax[0].set_title("image")

        ax[1].imshow(mark_boundaries(cancer_map, roi_mask, color=(1, 0, 0), ))
        im = ax[1].scatter(x, y, c=p, cmap='Spectral')
        ax[1].set_title("prob map")

        # fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()