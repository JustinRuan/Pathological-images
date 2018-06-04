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

class Test_detector(unittest.TestCase):

    def test_detect(self):
        dtor = Detector.Detector("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json",
                               "17004930 HE_2017-07-29 09_45_09.kfb",
                               '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930"
                               )

        roi_img = dtor.get_ROI_img(630, 630, 720, 720, 1, 5)
        dtor.set_ROI(630, 630, 720, 720, 1)

        seeds, tags = dtor.detect_ROI(20, 256)
        print(tags)

        result = dtor.draw_result(seeds,20,256,tags,630,630, 1)

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(roi_img)
        ax[0].set_title("roi_img")
        ax[1].imshow(result)
        ax[1].set_title("result")

        for a in ax.ravel():
            a.axis('off')

        plt.show()