#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""


import unittest
from core import *
import matplotlib.pyplot as plt
from pytorch.detector import Detector
import numpy as np

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"

class Test_detector(unittest.TestCase):

    def test_search_region(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        id = "003"
        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % id,
                                 'Tumor/tumor_%s.xml' % id, "Tumor_%s" % id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        x1 = 2400
        y1 = 4700
        x2 = 2600
        y2 = 4850

        # 001 的检测难度大，
        test_set = {"001" : (2100, 3800, 2400, 4000),
                    "003": (2400, 4700, 2600, 4850)}

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(mask_img)
        ax[1].set_title("mask_img")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return

    def test_detect2(self):

        test_set = {"001" : (2100, 3800, 2400, 4000),
                    "003": (2400, 4700, 2600, 4850)}
        id = "003"
        roi = test_set[id]
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % id,
                                 'Tumor/tumor_%s.xml' % id, "Tumor_%s" % id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        seeds, predictions = detector.detect_region(x1, y1, x2, y2, 1.25, 5, 128, interval = 64)
        new20_seeds, new20_predictions = detector.detect_region_detailed(seeds, predictions, 5, 128, 20, 256)
        new40_seeds, new40_predictions = detector.detect_region_detailed(new20_seeds, new20_predictions, 20, 256, 40, 256)
        # print(predictions_deep)

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        np.savez("{}/data/cancer_predictions_{}".format(c.PROJECT_ROOT, id), src_img,
                 seeds, predictions, new20_seeds, new20_predictions, new40_seeds, new40_predictions)

        # fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
        # ax = axes.ravel()
        #
        # ax[0].imshow(src_img)
        # ax[0].set_title("src_img")
        #
        # ax[1].imshow(src_img)
        # ax[1].imshow(cancer_map, alpha=0.6)
        # ax[1].set_title("cancer_map")
        #
        # ax[2].imshow(src_img)
        # ax[2].imshow(cancer_map2, alpha=0.6)
        # ax[2].set_title("cancer_map2")
        #
        # for a in ax.ravel():
        #     a.axis('off')
        #
        # plt.show()

        return


    def test_show_result(self):
        test_set = {"001" : (2100, 3800, 2400, 4000),
                    "003": (2400, 4700, 2600, 4850)}
        id = "003"
        roi = test_set[id]
        x1 = roi[0]
        y1 = roi[1]
        x2 = roi[2]
        y2 = roi[3]

        c = Params()
        c.load_config_file(JSON_PATH)

        D = np.load("{}/data/cancer_predictions_{}.npz".format(c.PROJECT_ROOT, id))
        src_img = D['arr_0']
        seeds = D['arr_1']
        predictions = D['arr_2']
        new20_seeds = D['arr_3']
        new20_predictions = D['arr_4']
        new40_seeds = D['arr_5']
        new40_predictions = D['arr_6']

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % id,
                                 'Tumor/tumor_%s.xml' % id, "Tumor_%s" % id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)
        detector.setting_detected_area(x1, y1, x2, y2, 1.25)
        cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 5, 1.25, seeds, predictions, 128,
                                                                     None, None)
        cancer_map2, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 20, 1.25, new20_seeds, new20_predictions,
                                                                      256, prob_map, count_map)
        cancer_map3, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 40, 1.25, new40_seeds, new40_predictions,
                                                                      256, prob_map, count_map)

        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        label_map = np.load("label_map.npy")
        cancer_map4 = detector.create_cancer_map_superpixels(cancer_map3, label_map)


        print("\n x5 低倍镜下的结果：")
        t1 = 0.8
        false_positive_rate_x5, true_positive_rate_x5, roc_auc_x5 = detector.evaluate(t1, cancer_map, mask_img)

        print("\n x20 高倍镜下增强的结果：")
        t2 = 0.8
        false_positive_rate_x20, true_positive_rate_x20, roc_auc_x20 = detector.evaluate(t2, cancer_map2, mask_img)

        print("\n x40 高倍镜下增强的结果：")
        t3 = 0.5
        false_positive_rate_x40, true_positive_rate_x40, roc_auc_x40 = detector.evaluate(t3, cancer_map3, mask_img)

        t4 = 0.5
        false_positive_rate_f, true_positive_rate_f, roc_auc_f = detector.evaluate(t4, cancer_map4, mask_img)

        fig, axes = plt.subplots(2, 4, figsize=(60, 40), dpi=100)
        ax = axes.ravel()

        ax[1].imshow(src_img)
        ax[1].set_title("src_img")

        ax[6].imshow(src_img)
        ax[6].imshow(mask_img, alpha=0.6)
        ax[6].set_title("mask_img")

        ax[0].set_title('Receiver Operating Characteristic')
        ax[0].plot(false_positive_rate_x5, true_positive_rate_x5, 'g',
                 label='x5  AUC = %0.4f' % roc_auc_x5)
        ax[0].plot(false_positive_rate_x20, true_positive_rate_x20, 'b',
                 label='x20 AUC = %0.4f' % roc_auc_x20)
        ax[0].plot(false_positive_rate_x40, true_positive_rate_x40, 'c',
                 label='x40 AUC = %0.4f' % roc_auc_x40)
        ax[0].plot(false_positive_rate_f, true_positive_rate_f, 'r',
                 label='final AUC = %0.4f' % roc_auc_f)

        ax[0].legend(loc='lower right')
        ax[0].plot([0, 1], [0, 1], 'r--')
        ax[0].set_xlim([-0.1, 1.2])
        ax[0].set_ylim([-0.1, 1.2])
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_xlabel('False Positive Rate')

        ax[2].imshow(src_img)
        ax[2].imshow(cancer_map, alpha=0.6)
        ax[2].contour(cancer_map >= t1)
        ax[2].set_title("cancer_map, t = %s" % t1)

        ax[3].imshow(src_img)
        ax[3].imshow(cancer_map2, alpha=0.6)
        ax[3].contour(cancer_map >= t2)
        ax[3].set_title("cancer_map2, t = %s" % t2)

        ax[7].imshow(src_img)
        ax[7].imshow(cancer_map3, alpha=0.6)
        ax[7].contour(cancer_map3 >= t3)
        ax[7].set_title("cancer_map3, t = %s" % t3)

        ax[5].imshow(src_img)
        ax[5].imshow(cancer_map4, alpha=0.6)
        ax[5].contour(cancer_map4>t4)
        ax[5].set_title("final cancer_map, t = %s" % t4)

        for a in ax.ravel():
            a.axis('off')
        ax[0].axis("on")

        plt.show()


