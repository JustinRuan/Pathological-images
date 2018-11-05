#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

from skimage import color
import numpy as np

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
        lsbO = np.std(labO_l)
        asbO = np.std(labO_a)
        bsbO = np.std(labO_b)

        lMO = np.mean(labO_l)
        aMO = np.mean(labO_a)
        bMO = np.mean(labO_b)

        labO_l= (labO_l - lMO) / lsbO
        labO_a = (labO_a - aMO) / asbO
        labO_b = (labO_b - bMO) / bsbO

        # labO_l[labO_l > 4] = 4
        # labO_l[labO_l < -4] = -4
        # labO_a[labO_a > 8] = 8
        # labO_a[labO_a < -8] = -8
        # labO_b[labO_b > 8] = 8
        # labO_b[labO_b < -8] = -8

        labO_ls = 100 * (labO_l + 4) / 8
        labO_as = 40 * labO_a
        labO_bs = 40 * labO_b
        labO = np.dstack([labO_ls, labO_as, labO_bs])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)

        return rgb_image