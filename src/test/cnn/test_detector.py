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
from skimage.color import label2rgb
from skimage.measure import label

class Test_detector(unittest.TestCase):

    def test_detect(self):
        dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
                               "17004930 HE_2017-07-29 09_45_09.kfb",
                               '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
                               )
        x1 = 600
        y1 = 610
        x2 = 720
        y2 = 720
        roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 5)
        dtor.set_ROI(x1, y1, x2, y2, 1)

        seeds, tags = dtor.detect_ROI(20, 256)
        # print(tags)

        result = dtor.draw_result(seeds,20,256,tags,x1,y1, 1)

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(roi_img)
        ax[0].set_title("roi_img")

        # img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 1.25)
        # ax[1].imshow(img)
        ax[1].imshow(result, alpha=1)
        ax[1].set_title("result")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

    def test_segment(self):
        dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
                               "17004930 HE_2017-07-29 09_45_09.kfb",
                               '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
                               )
        x1 = 600
        y1 = 610
        x2 = 750
        y2 = 750
        roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 5)
        segments = dtor.segment_image(roi_img, 10)
        print(np.max(segments))

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(roi_img)
        ax[0].set_title("roi_img")
        ax[1].imshow(segments)
        ax[1].set_title("segments")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

    def test_detect_ROI_regions(self):
        dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
                               "17004930 HE_2017-07-29 09_45_09.kfb",
                               '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
                               )
        x1 = 600
        y1 = 600
        x2 = 720
        y2 = 720
        roi_img = dtor.get_ROI_img(x1, y1, x2, y2, 1, 1.25)
        dtor.set_ROI(x1, y1, x2, y2, 1)

        regions,result = dtor.detect_ROI_regions(x1, y1, x2, y2, 1, 50, 20, 256)

        fig, axes = plt.subplots(2, 2, figsize=(4, 4), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(roi_img)
        ax[0].set_title("roi_img")

        # label image regions
        label_image = label(regions)
        image_label_overlay = label2rgb(label_image, alpha=0.3, image=roi_img)
        ax[1].imshow(image_label_overlay)
        ax[1].set_title("regions")

        ax[2].imshow(result, alpha=1)
        ax[2].set_title("result")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return