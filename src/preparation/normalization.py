#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

import time
from skimage import color
import numpy as np
from skimage import io

class ImageNormalization(object):
    def __init__(self, params):
        self._params = params
        return

    def calculate_avg_mean_std(self, data_filename):
        root_path = self._params.PATCHS_ROOT_PATH
        data_file = "{}/{}".format(root_path, data_filename)

        count = 0
        mean_l = []
        mean_a = []
        mean_b = []
        std_l = []
        std_a = []
        std_b = []
        f = open(data_file, "r")
        for line in f:
            items = line.split(" ")
            patch_file = "{}/{}".format(root_path, items[0])
            img = io.imread(patch_file, as_gray=False)

            lab_img = color.rgb2lab(img)

            # LAB三通道分离
            labO_l = lab_img[:, :, 0]
            labO_a = lab_img[:, :, 1]
            labO_b = lab_img[:, :, 2]

            # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
            std_l.append(np.std(labO_l))
            std_a.append(np.std(labO_a))
            std_b.append(np.std(labO_b))

            mean_l.append(np.mean(labO_l))
            mean_a.append(np.mean(labO_a))
            mean_b.append(np.mean(labO_b))

            if (0 == count%1000):
                print("{} calculate mean and std >>> {}".format(time.asctime( time.localtime()), count))
            count += 1

        f.close()

        avg_mean_l = np.mean(mean_l)
        avg_mean_a = np.mean(mean_a)
        avg_mean_b = np.mean(mean_b)
        avg_std_l = np.mean(std_l)
        avg_std_a = np.mean(std_a)
        avg_std_b = np.mean(std_b)

        return avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b

    '''
    Lab颜色空间中的L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
    a表示从红色到绿色的范围，取值范围是[127,-128]；
    b表示从黄色到蓝色的范围，取值范围是[127,-128]。
    '''
    @staticmethod
    def normalize(src_img, avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b):
     #   avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b = \
     #       64.4142491342, 17.8558880865, -14.9493230523, 9.69010566298, 4.87607967592, 4.22826851423  # 5 x 128
    # 62.8356120408,19.4081460042,-16.2194449226,12.0661142823,6.86495094844,7.14640136961   # 20 x 256
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        labO_l = lab_img[:, :, 0]
        labO_a = lab_img[:, :, 1]
        labO_b = lab_img[:, :, 2]

        # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
        lsbO = np.std(labO_l)
        asbO = np.std(labO_a)
        bsbO = np.std(labO_b)

        lMO = np.mean(labO_l)
        aMO = np.mean(labO_a)
        bMO = np.mean(labO_b)

        labO_l= (labO_l - lMO) / lsbO * avg_std_l + avg_mean_l
        labO_a = (labO_a - aMO) / asbO * avg_std_a + avg_mean_a
        labO_b = (labO_b - bMO) / bsbO * avg_std_b + avg_mean_b

        # labO_l[labO_l > 4] = 4
        # labO_l[labO_l < -4] = -4
        # labO_a[labO_a > 8] = 8
        # labO_a[labO_a < -8] = -8
        # labO_b[labO_b > 8] = 8
        # labO_b[labO_b < -8] = -8

        # labO_ls = 100 * (labO_l + 4) / 8
        # labO_as = 40 * labO_a
        # labO_bs = 40 * labO_b
        # labO = np.dstack([labO_ls, labO_as, labO_bs])

        labO = np.dstack([labO_l, labO_a, labO_b])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)

        return rgb_image