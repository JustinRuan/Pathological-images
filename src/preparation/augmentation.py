#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-13'

"""
import time
import os
from skimage import color
import numpy as np
from skimage import io
import random

class ImageAugmentation(object):

    def __init__(self, **kwarg):
        # 实验2用：l_range = (0.9, 1.1), a_range = (0.95, 1.05), b_range = (0.95, 1.05), constant_range = (-0.5, 0.5)
        # 实验1用：l_range = (0.99, 1.01), a_range = (0.99, 1.01), b_range = (0.99, 1.01), constant_range = (-5, 5)
        self.l_range = kwarg["l_range"]
        self.a_range = kwarg["a_range"]
        self.b_range = kwarg["b_range"]
        self.constant_range = kwarg["constant_range"]

    # def augment_images(self, src_img):
    #     lab_img = color.rgb2lab(src_img)
    #
    #     # LAB三通道分离
    #     labO_l = np.array(lab_img[:, :, 0])
    #     labO_a = np.array(lab_img[:, :, 1])
    #     labO_b = np.array(lab_img[:, :, 2])
    #
    #     # randomly modify the lab space to do color augmentation
    #     f1_l = random.uniform(self.l_range[0], self.l_range[1])
    #     f1_c = random.uniform(self.constant_range[0], self.constant_range[1])
    #     lab1_l = f1_l * labO_l +  f1_c # Adjust Luminance
    #
    #     f2_a = random.uniform(self.a_range[0], self.a_range[1])
    #     f2_c = random.uniform(self.constant_range[0], self.constant_range[1])
    #     lab1_a = f2_a * labO_a + f2_c  # Adjust color
    #
    #     f3_b = random.uniform(self.b_range[0], self.b_range[1])
    #     f3_c = random.uniform(self.constant_range[0], self.constant_range[1])
    #     lab1_b = f3_b * labO_b + f3_c  # Adjust color
    #
    #     # for debug
    #     # print("Random state: ", f1_l, f1_c, f2_a, f2_c, f3_b, f3_c)
    #
    #     lab2_l = np.clip(lab1_l, 0, 100)
    #     lab2_a = np.clip(lab1_a, -128, 127)
    #     lab2_b = np.clip(lab1_b, -128, 127)
    #
    #     labO = np.dstack([lab2_l, lab2_a, lab2_b])
    #     # LAB to RGB变换
    #     rgb_image = color.lab2rgb(labO)
    #     return rgb_image

    def augment_images(self, src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        labO_l = np.array(lab_img[:, :, 0])
        labO_a = np.array(lab_img[:, :, 1])
        labO_b = np.array(lab_img[:, :, 2])

        tag = labO_l <= 75
        # print(np.sum(tag))

        # randomly modify the lab space to do color augmentation
        f1_l = random.uniform(self.l_range[0], self.l_range[1])
        f1_c = random.uniform(self.constant_range[0], self.constant_range[1])
        labO_l[tag] = f1_l * labO_l[tag] +  f1_c # Adjust Luminance

        f2_a = random.uniform(self.a_range[0], self.a_range[1])
        f2_c = random.uniform(self.constant_range[0], self.constant_range[1])
        labO_a[tag] = f2_a * labO_a[tag] + f2_c  # Adjust color

        f3_b = random.uniform(self.b_range[0], self.b_range[1])
        f3_c = random.uniform(self.constant_range[0], self.constant_range[1])
        labO_b[tag] = f3_b * labO_b[tag] + f3_c  # Adjust color

        # for debug
        # print("Random state: ", f1_l, f1_c, f2_a, f2_c, f3_b, f3_c)

        lab2_l = np.clip(labO_l, 0, 100)
        lab2_a = np.clip(labO_a, -128, 127)
        lab2_b = np.clip(labO_b, -128, 127)

        labO = np.dstack([lab2_l, lab2_a, lab2_b])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)
        return rgb_image



