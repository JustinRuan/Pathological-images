#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-13'

"""

import unittest
from core import *
import matplotlib.pyplot as plt


class TestOpen_Slide(unittest.TestCase):

    def test_load(self):
        slide = Open_Slide()
        tag = slide.open_slide("D:/Data/CAMELYON16/Tumor/Tumor_001.tif","Tumor_001")
        if tag:
            scale = 5
            w , h = slide.get_image_width_height_byScale(40)
            print(w, h)
            print(slide.img.level_dimensions)
            print(slide.img.level_downsamples)
            img = slide.get_thumbnail(1.25)
            img2 = slide.get_image_block(5, 6600, 16000, 200, 200)

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

    def test_read_annotation(self):
        slide = Open_Slide()
        slide.read_annotation("D:/Data/CAMELYON16/Tumor/tumor_005.xml")
