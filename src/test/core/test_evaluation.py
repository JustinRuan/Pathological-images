#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-12'

"""

import os
import unittest
from core import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage import color

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"


class TestEvaluation(unittest.TestCase):
    def test_save_result_xml(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        i = 9
        code = "Tumor_{:0>3d}".format(i)
        filename = "{}/results/{}_cancermap.npz".format( c.PROJECT_ROOT, code)
        result = np.load(filename)
        x1 = result["x1"]
        y1 = result["y1"]
        coordinate_scale = result["scale"]
        cancer_map = result["cancer_map"]

        # levels = [0.2, 0.3, 0.5, 0.6, 0.8]
        levels = [0.5]
        eval = Evaluation(c)
        eval.save_result_xml(code, x1, y1, coordinate_scale, cancer_map, levels)


    def test_create_true_mask_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)

        # eval.create_true_mask_file("Train_Tumor", np.arange(100, 112))
        eval.create_true_mask_file("testing\images", [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,
                                                      61,64,65,66,68,69,71,73,74,75,79,
                                                    82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122])

    def test_show_mask(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        mask_folder = "{}/data/true_masks".format(c.PROJECT_ROOT)
        index = 11
        maskDIR = "{}/Tumor_{:0>3d}_true_mask.npy".format(mask_folder, index)
        L0_RESOLUTION, EVALUATION_MASK_LEVEL = 0.243, 5

        evaluation_mask = eval.computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        print("Max label :", np.max(evaluation_mask))

        plt.imshow(color.label2rgb(evaluation_mask, bg_label=0))
        plt.show()


    def test_calculate_ROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)

        tag = "Tumor"
        # tag = "Test"
        # tag = "Test_Normal"
        # tag = "Normal"

        if tag == "Tumor":
            # select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)]
            select = ["Tumor_{:0>3d}".format(i) for i in [20,29,33,61,89,95]]
            eval.calculate_ROC("Train_Tumor", tag=64, chosen=select, p_thresh=0.5)
        elif tag == "Test":

            select = ["Test_{:0>3d}".format(i) for i in [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,
                                                          61,64,65,66,68,69,71,73,74,75,79,
                                                        82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]]
            # select = ["Test_{:0>3d}".format(i) for i in [4,10,29,30,33,38,48,66,79,84,99,102,116,117,122]]
            eval.calculate_ROC("testing\images", tag=64, chosen=select, p_thresh=0.5)
        elif tag == "Test_Normal":
            temp = [3, 5, 6, 7, 9, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 28, 31, 32, 34, 35, 36, 37, 39, 41, 42, 43, 44, 45,
             47, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 67, 70, 72, 76, 77, 78, 80, 81, 83, 85, 86, 87, 88, 89, 91,
             92, 93, 95, 96, 98, 100, 101, 103, 106, 107, 109, 111, 112, 114, 115, 118, 119, 120, 123, 124, 125, 126, 127,
             128, 129, 130]

            select = ["Test_{:0>3d}".format(i) for i in temp]
            eval.calculate_ROC("testing\images", tag=64, chosen=select, p_thresh=0.5)
        elif tag == "Normal":
            # select = ["Normal_{:0>3d}".format(i) for i in range(1,161)]
            select = ["Normal_{:0>3d}".format(i) for i in range(61, 161)]
            eval.calculate_ROC("Train_Normal", tag=0, chosen=select, p_thresh=0.5)


    def test_save_result_pictures(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        # eval.save_result_pictures("Train_Tumor", chosen=["Tumor_002", "Tumor_003"])
        # eval.save_result_pictures("Train_Tumor", chosen=None)
        # select = ["Tumor_{:0>3d}".format(i) for i in range(25, 51)]
        # select = ["Tumor_{:0>3d}".format(i) for i in [98]]
        # eval.save_result_pictures("Train_Tumor", tag = 0,  chosen=select)

        select = ["Test_{:0>3d}".format(i) for i in [1]]
        eval.save_result_pictures("testing\images", tag=64, chosen=select)

    def test_evaluation_FROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)

        mask_folder = "{}/data/true_masks".format(c.PROJECT_ROOT)
        result_folder = "{}/results/csv_2".format(c.PROJECT_ROOT)

        eval.evaluation_FROC(mask_folder, result_folder, level = 7)
