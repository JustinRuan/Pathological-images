#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""


import unittest
from core import *
import matplotlib.pyplot as plt
from cnn import *
import numpy as np
from transfer import Transfer

class Test_detector(unittest.TestCase):

    def test_detect(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")

        detector = Detector(c, imgCone)
        # print(detector.ImageHeight, detector.ImageWidth)

        x1 = 600
        y1 = 610
        x2 = 1420
        y2 = 1420
        seeds, predictions = detector.detect_region(x1, y1, x2, y2, 1, 5, 128)
        cancer_map, prob_map, count_map = detector.create_cancer_map(x1,y1, 1, 5, 1.25, seeds, predictions, 128, None, None)
        # print(cancer_map)

        src_img = detector.get_detect_area_img(x1, y1, x2, y2, 1, 1.25)

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(src_img)
        ax[1].imshow(cancer_map, alpha=0.6)
        ax[1].set_title("cancer_map")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return


    def test_detect2(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")
        imgCone = ImageCone(c)

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")

        detector = Detector(c, imgCone)
        # print(detector.ImageHeight, detector.ImageWidth)

        x1 = 600
        y1 = 610
        x2 = 720
        y2 = 720
        seeds, predictions = detector.detect_region(x1, y1, x2, y2, 1, 5, 128)
        new_seeds = detector.get_seed_deep_analysis(seeds, predictions, 5, 128, 20, 256)

        cnn = Transfer(c)
        predictions_deep = cnn.predict(imgCone, 20, 256, new_seeds)
        # print(result)

        cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1, 5, 1.25, seeds, predictions, 128, None,
                                                                     None)
        cancer_map2, prob_map, count_map = detector.create_cancer_map(x1, y1, 1, 20, 1.25, new_seeds, predictions_deep,
                                                                      256, prob_map, count_map)
        # cancer_map2, prob_map, count_map = detector.create_cancer_map(x1, y1, 1, 20, 1.25, new_seeds, predictions_deep,
        #                                                               256, None, None)
        # print(cancer_map)

        src_img = detector.get_detect_area_img(x1, y1, x2, y2, 1, 1.25)

        fig, axes = plt.subplots(1, 3, figsize=(8, 6), dpi=200)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        ax[0].set_title("src_img")

        ax[1].imshow(src_img)
        ax[1].imshow(cancer_map, alpha=0.6)
        ax[1].set_title("cancer_map")

        ax[2].imshow(src_img)
        ax[2].imshow(cancer_map2, alpha=0.6)
        ax[2].set_title("cancer_map2")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return
    # def test_detect(self):
    #     dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
    #                            "17004930 HE_2017-07-29 09_45_09.kfb",
    #                            '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
    #                            )
    #     x1 = 600
    #     y1 = 610
    #     x2 = 720
    #     y2 = 720
    #     roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 5)
    #     dtor.set_ROI(x1, y1, x2, y2, 1)
    #
    #     seeds, tags = dtor.detect_ROI(20, 256)
    #     # print(tags)
    #
    #     result = dtor.draw_result(seeds,20,256,tags,x1,y1, 1)
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
    #     ax = axes.ravel()
    #
    #     ax[0].imshow(roi_img)
    #     ax[0].set_title("roi_img")
    #
    #     # img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 1.25)
    #     # ax[1].imshow(img)
    #     ax[1].imshow(result, alpha=1)
    #     ax[1].set_title("result")
    #
    #     for a in ax.ravel():
    #         a.axis('off')
    #
    #     plt.show()
    #
    # def test_segment(self):
    #     dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
    #                            "17004930 HE_2017-07-29 09_45_09.kfb",
    #                            '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
    #                            )
    #     x1 = 600
    #     y1 = 610
    #     x2 = 750
    #     y2 = 750
    #     roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 5)
    #     segments = dtor.segment_image(roi_img,30)
    #     print(np.max(segments))
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
    #     ax = axes.ravel()
    #
    #     ax[0].imshow(roi_img)
    #     ax[0].set_title("roi_img")
    #     ax[1].imshow(segments)
    #     ax[1].set_title("segments")
    #
    #     for a in ax.ravel():
    #         a.axis('off')
    #
    #     plt.show()
    #
    # def test_detect_ROI_regions(self):
    #     dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
    #                            "17004930 HE_2017-07-29 09_45_09.kfb",
    #                            '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
    #                            )
    #     x1 = 600
    #     y1 = 600
    #     x2 = 1420
    #     y2 = 1420
    #     roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 1.25)
    #     dtor.set_ROI(x1, y1, x2, y2, 1)
    #
    #     regions,result = dtor.detect_ROI_regions(x1, y1, x2, y2, 1, 100, 20, 256)
    #
    #     fig, axes = plt.subplots(2, 2, figsize=(4, 4), dpi=300)
    #     ax = axes.ravel()
    #
    #     ax[0].imshow(roi_img)
    #     ax[0].set_title("roi_img")
    #
    #     # label image regions
    #     label_image = label(regions)
    #     image_label_overlay = label2rgb(label_image, alpha=0.3, image=roi_img)
    #     ax[1].imshow(image_label_overlay)
    #     ax[1].set_title("regions")
    #
    #     ax[2].imshow(result, alpha=1)
    #     ax[2].set_title("result")
    #
    #     for a in ax.ravel():
    #         a.axis('off')
    #
    #     plt.show()
    #
    #     return
    #
    # def test_test02(self):
    #     myList = []
    #     myList.append((1,3))
    #     myList.append((2,4))
    #     myList.append((3,7))
    #
    #     x = (np.array(myList))[:,0]
    #     print(x)


