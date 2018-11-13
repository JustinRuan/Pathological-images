#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-12'

"""

import unittest
from core import *
import matplotlib.pyplot as plt


class TestASAP_Slide(unittest.TestCase):

    def test_load(self):
        slide = ASAP_Slide()
        tag = slide.open_slide("D:/Data/CAMELYON16/Tumor/Tumor_001.tif","Tumor_001")
        if tag:
            scale = 5
            w , h = slide.get_image_width_height_byScale(1)
            print(w, h)
            img = slide.get_image_block(2, 20000, 20000, 1000, 1000)
            img2 = slide.get_image_block(4, 20000, 40000, 2000, 2000)

            # img = slide.get_image_block(scale, w>>1, h>>1, 1000, 1000)

            fig, axes = plt.subplots(1, 2, figsize=(4, 3), dpi=200)
            ax = axes.ravel()

            ax[0].imshow(img)
            ax[0].set_title("img")

            ax[1].imshow(img2)
            ax[1].set_title("img2")

            for a in ax.ravel():
                a.axis('off')

            plt.show()

        return