#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

from skimage import color
import numpy as np
from sklearn import preprocessing

class ImageNormalization(object):
    def __init__(self):

        return

    '''
    Lab颜色空间中的L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
    a表示从红色到绿色的范围，取值范围是[127,-128]；
    b表示从黄色到蓝色的范围，取值范围是[127,-128]。
    '''
    @staticmethod
    def normalize(src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        labO_l = lab_img[:, :, 0]
        labO_a = lab_img[:, :, 1]
        labO_b = lab_img[:, :, 2]

        # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差: [-2, 2]
        labO_l = preprocessing.scale(labO_l)
        labO_a = preprocessing.scale(labO_a)
        labO_b = preprocessing.scale(labO_b)
        labO_l[labO_l > 2] = 2
        labO_l[labO_l < -2] = -2
        labO_a[labO_a > 2] = 2
        labO_a[labO_a < -2] = -2
        labO_b[labO_b > 2] = 2
        labO_b[labO_b < -2] = -2

        labO_ls = 25 * labO_l + 50       # 100 * (labO_l + 2)/ 4
        labO_as = 64 * labO_a
        labO_bs = 64 * labO_b
        labO = np.dstack([labO_ls, labO_as, labO_bs])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)
        return rgb_image