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

    # def test_search_max_points(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     i = 9
    #     code = "Tumor_{:0>3d}".format(i)
    #     filename = "{}/results/{}_cancermap.npz".format( c.PROJECT_ROOT, code)
    #     result = np.load(filename)
    #     x1 = result["x1"]
    #     y1 = result["y1"]
    #     coordinate_scale = result["scale"]
    #     cancer_map = result["cancer_map"]
    #
    #     # thresh_list = [0.6, 0.5, 0.4, 0.3, 0.2]
    #     thresh_list = [0.5]
    #     eval = Evaluation(c)
    #     result = eval.search_local_max_points(cancer_map, thresh_list, x1, y1)
    #
    #     imgCone = ImageCone(c, Open_Slide())
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("Train_Tumor/%s.tif" % code,
    #                              'Train_Tumor/%s.xml' % code, code)
    #
    #     src_img = imgCone.get_fullimage_byScale(1.25)
    #     mask_img = imgCone.create_mask_image(1.25, 0)
    #     mask_img = mask_img['C']
    #
    #     x = []
    #     y = []
    #     p = []
    #     for item in result:
    #         x.append(item['x'] // 32)
    #         y.append(item['y'] // 32)
    #         p.append(item['prob'])
    #
    #     plt.figure()
    #     plt.imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
    #     plt.scatter(x, y, c = p,cmap='Spectral')
    #     plt.colorbar()
    #     plt.show()


    # def test_output_result_csv(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #     eval = Evaluation(c)
    #     # eval.output_result_csv(["Tumor_009", "Tumor_011", "Tumor_016", "Tumor_026", "Tumor_039"])
    #     # eval.output_result_csv(["Tumor_009"])
    #     # eval.output_result_csv([0.5], ["Tumor_009"])
    #     eval.output_result_csv([0.5, 0.35], None)