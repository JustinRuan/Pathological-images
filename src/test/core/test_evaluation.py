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

        eval.create_true_mask_file("Train_Tumor", np.arange(12, 58))

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

    def test_evaluation_FROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)

        mask_folder = "{}/data/true_masks".format(c.PROJECT_ROOT)
        result_folder = "{}/results/csv_2".format(c.PROJECT_ROOT)

        eval.evaluation_FROC(mask_folder, result_folder)

    def test_calculate_ROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        # , "Tumor_034", "Tumor_035", "Tumor_036"
        # eval.calculate_ROC("Train_Tumor", chosen=["Tumor_040", "Tumor_041",])
        # eval.calculate_ROC("Train_Tumor", chosen=None)
        eval.calculate_ROC("Train_Tumor", chosen=["Tumor_017", ])

    def test_save_result_pictures(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        eval.save_result_pictures("Train_Tumor", chosen=["Tumor_001", ])