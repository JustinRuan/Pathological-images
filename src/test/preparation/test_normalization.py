#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
from skimage.io import imread
import random
import numpy as np

from preparation.normalization import RGBNormalization, ReinhardNormalization, HistNormalization, ImageNormalizationTool
# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestNormalization(unittest.TestCase):

    def test_normalization(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        # filename = "T_NC_Simple0330_4000_256_test.txt"
        filename = "T_NC_P0404_4000_256_test.txt"
        patch_path = c.PATCHS_ROOT_PATH["P0404"]
        all_file_list, _ = util.read_csv_file(patch_path, "{}/{}".format(patch_path, filename))

        N = 20
        rnd = random.randint(0, len(all_file_list) // N)
        file_list = all_file_list[rnd:rnd + N]

        # normal = ReinhardNormalization("reinhard", target_mean=(72.66, 16.89, -9.979),
        #                             target_std=(13.42, 5.767, 4.891),
        #                             source_mean=(72.45, 17.63, -17.77),
        #                             source_std=(10.77, 7.064, 6.50))

        # normal = RGBNormalization("rgb_norm", target_mean=(198.9, 168.0, 206.2),
        #                             target_std=(27.94, 31.93, 22.56),
        #                             source_mean=(194.1, 169.3, 210.4),
        #                             source_std=(27.86, 30.92, 20.25))

        normal = HistNormalization("match_hist", hist_target = "hist_templates.npy",
                                    hist_source = "hist_soures.npy")

        # image_list = []
        # for filename in file_list:
        #     img = imread(filename)
        #     image_list.append(img)
        #
        # normal = HistNormalization("match_hist", hist_target ="hist_templates.npy",
        #                            hist_source = None)
        # normal.prepare(image_list)

        fig = plt.figure(figsize=(16, 10), dpi=100)
        for index, filename in enumerate(file_list):
            img = imread(filename)
            result = normal.normalize(img)
            plt.subplot(5, 2 * N / 5, 2 * index + 1)
            plt.imshow(img)

            plt.axis("off")
            plt.subplot(5, 2 * N / 5, 2 * index + 2)
            plt.axis("off")
            plt.imshow(result)

        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        plt.show()

    def test_calculate_mean_std(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = ImageNormalizationTool(c)
        filename = "T_NC_P0404_4000_256_test.txt"
        avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b = \
            normal.calculate_avg_mean_std([filename])

        print(avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b)
        # P0327 Test
        # 72.6668955191 16.8991760939 -9.97997070951 13.427538741 5.76732834018 4.89131967314
        # (72.66, 16.89, -9.979), (13.42, 5.767, 4.891)

        # T_NC_P0404_4000_256.txt
        # 72.4550587842 17.6308215754 -17.7722937628 10.7780328825 7.06444637399 6.50070533929
        # (72.45, 17.63, -17.77), (10.77, 7.064, 6.50)

    def test_calculate_mean_std_RGB(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = ImageNormalizationTool(c)
        filename = "T_NC_P0404_4000_256.txt"
        # filename = "T_NC_Simple0327_2_4000_256_test.txt"
        avg_mean_r, avg_mean_g, avg_mean_b, avg_std_r, avg_std_g, avg_std_b = \
            normal.calculate_avg_mean_std_RGB([filename])

        print(avg_mean_r, avg_mean_g, avg_mean_b, avg_std_r, avg_std_g, avg_std_b)
        # T_NC_Simple0327_2_4000_256_test
        #  0.788584186842 0.666972466448 0.775474451527 0.132370413651 0.147355493385 0.117168562872
        # (0.7886, 0.6670, 0.7755), (0.1324, 0.1474, 0.1172)
        # 198.981852223 168.015340194 206.203351887 27.940084234 31.9342429447 22.5670791219
        # (198.9, 168.0, 206.2) (27.94, 31.93, 22.56)

        # T_NC_Simple0330_4000_256_test.txt
        # 0.780320989111 0.658883687035 0.808640595636 0.10956895778 0.125232325273 0.0884983494975
        # (0.7803, 0.6589, 0.8086), (0.1096, 0.1252, 0.0885)
        # 201.200932593 170.389103839 197.969921493 33.6008670407 37.3686477541 29.7284223274
        # (201.2, 170.3, 197.9) (33.6, 37.36, 29.72)

        # T_NC_P0404_4000_256.txt
        # 194.179581005 169.338380381 210.455308373 27.8654252516 30.9280496823 20.2526431351
        # (194.1, 169.3, 210.4) (27.86, 30.92, 20.25)

    def test_calculate_mean_std_HSD(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = ImageNormalizationTool(c)

        # avg_mean_h, avg_mean_s, avg_mean_d, avg_std_h, avg_std_s, avg_std_d = \
        #     normal.calculate_avg_mean_std_HSD("P0327", ["T_NC_Simple0327_2_4000_256_test.txt"])

        avg_mean_h, avg_mean_s, avg_mean_d, avg_std_h, avg_std_s, avg_std_d = \
            normal.calculate_avg_mean_std_HSD("P0330", ["T_NC_Simple0330_4000_256_test.txt"])

        print(avg_mean_h, avg_mean_s, avg_mean_d, avg_std_h, avg_std_s, avg_std_d)
        # "T_NC_Simple0327_2_4000_256_test.txt"
        # -0.257421384092 0.235364054717 0.389308679108 0.18601853328 0.188481962243 0.248231192375

        # T_NC_Simple0404_4000_256_test.txt
        # -0.0676884085515 0.408808235442 0.371045973718 0.125415498138 0.124773051469 0.19882963109

        # T_NC_Simple0330_4000_256_test.txt
        # -0.163529218718 0.35086233971 0.375252081355 0.151278120093 0.15350647589 0.209511975401

    def test_calc_hist(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = ImageNormalizationTool(c)
        normal.calculate_hist("P0327", "T_NC_Simple0327_2_4000_256_test.txt", file_code = "hist_soures_P0327")
        # normal.calculate_hist("Target", "Target_T1_4000_256_test.txt", file_code = "hist_templates")
        # normal.calculate_hist("P0330", "T_NC_Simple0330_4000_256_test.txt", "Target", "Target_T1_4000_256_test.txt")

    def test_draw_hist(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        normal = HistNormalization("match_hist", hist_target ="hist_templates_P0404.npy",
                                   hist_source = "hist_soures_P0404.npy",
                                   image_source= None)
        # normal.draw_hist("Nice")
        normal.draw_normalization_func("Nice")
