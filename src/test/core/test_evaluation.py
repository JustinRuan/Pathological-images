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

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"


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

    def test_evaluation_FROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)

        mask_folder = "{}/data/true_masks".format(c.PROJECT_ROOT)
        result_folder = "{}/results/csv_1".format(c.PROJECT_ROOT)

        eval.evaluation_FROC(mask_folder, result_folder)





    def test_calculate_ROC(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        # , "Tumor_034", "Tumor_035", "Tumor_036"

        # select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)]
        # # select = ["Tumor_{:0>3d}".format(i) for i in [98]]
        # eval.calculate_ROC("Train_Tumor", tag=256, chosen=select)

        # # eval.calculate_ROC("Train_Tumor", tag=2, chosen=None)

        select = ["Test_{:0>3d}".format(i) for i in [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,
                                                      61,64,65,66,68,69,71,73,74,75,79,
                                                    82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]]
        # # select = ["Test_{:0>3d}".format(i) for i in [4,10,29,30,33,38,48,66,79,84,99,102,116,117,122]]
        # select = ["Test_{:0>3d}".format(i) for i in [99, 117]]
        eval.calculate_ROC("testing\images", tag=128, chosen=select)

    def test_save_result_pictures(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        eval = Evaluation(c)
        # eval.save_result_pictures("Train_Tumor", chosen=["Tumor_002", "Tumor_003"])
        # eval.save_result_pictures("Train_Tumor", chosen=None)
        # select = ["Tumor_{:0>3d}".format(i) for i in range(25, 51)]
        select = ["Tumor_{:0>3d}".format(i) for i in [98]]
        eval.save_result_pictures("Train_Tumor", tag = 0,  chosen=select)

        # select = ["Test_{:0>3d}".format(i) for i in [13]]
        # eval.save_result_pictures("testing\images", tag=0, chosen=select)