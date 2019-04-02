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

# Reinhard algorithm
class ImageNormalization(object):
    def __init__(self, params, avg_mean_l=0, avg_mean_a=0, avg_mean_b=0, avg_std_l=0, avg_std_a=0, avg_std_b=0):
        self._params = params
        self.avg_mean_l = avg_mean_l
        self.avg_mean_a = avg_mean_a
        self.avg_mean_b = avg_mean_b
        self.avg_std_l = avg_std_l
        self.avg_std_a = avg_std_a
        self.avg_std_b = avg_std_b
        return

    '''
    Lab颜色空间中的L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
    a表示从红色到绿色的范围，取值范围是[127,-128]；
    b表示从黄色到蓝色的范围，取值范围是[127,-128]。
    '''
    def calculate_avg_mean_std(self, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH

        count = 0
        mean_l = []
        mean_a = []
        mean_b = []
        std_l = []
        std_a = []
        std_b = []

        for data_filename in data_filenames:
            data_file = "{}/{}".format(root_path, data_filename)


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

    def normalize(self, src_img):
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

        labO_l= (labO_l - lMO) / lsbO * self.avg_std_l + self.avg_mean_l
        labO_a = (labO_a - aMO) / asbO * self.avg_std_a + self.avg_mean_a
        labO_b = (labO_b - bMO) / bsbO * self.avg_std_b + self.avg_mean_b

        labO_l[labO_l > 100] = 100
        labO_l[labO_l < 0] = 0
        labO_a[labO_a > 127] = 127
        labO_a[labO_a < -128] = -128
        labO_b[labO_b > 127] = 127
        labO_b[labO_b < -128] = -128

        labO = np.dstack([labO_l, labO_a, labO_b])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)

        return rgb_image

    # @staticmethod
    # def normalize_shift_mean(src_img, avg_mean_l, avg_mean_a, avg_mean_b):
    #     lab_img = color.rgb2lab(src_img)
    #
    #     # LAB三通道分离
    #     labO_l = lab_img[:, :, 0]
    #     labO_a = lab_img[:, :, 1]
    #     labO_b = lab_img[:, :, 2]
    #
    #     # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
    #     lsbO = np.std(labO_l)
    #     asbO = np.std(labO_a)
    #     bsbO = np.std(labO_b)
    #
    #     lMO = np.mean(labO_l)
    #     aMO = np.mean(labO_a)
    #     bMO = np.mean(labO_b)
    #
    #     # labO_l= (labO_l - lMO) / lsbO * avg_std_l + avg_mean_l
    #     # labO_a = (labO_a - aMO) / asbO * avg_std_a + avg_mean_a
    #     # labO_b = (labO_b - bMO) / bsbO * avg_std_b + avg_mean_b
    #
    #     labO_l = (labO_l - lMO) + avg_mean_l
    #     labO_a = (labO_a - aMO) + avg_mean_a
    #     labO_b = (labO_b - bMO) + avg_mean_b
    #
    #     labO = np.dstack([labO_l, labO_a, labO_b])
    #     # LAB to RGB变换
    #     rgb_image = color.lab2rgb(labO)
    #
    #     return rgb_image


class ImageNormalization_RGB(object):
    def __init__(self, params):
        self._params = params
        # 归一化时，使用的参数
        return

    def calculate_avg_mean_std_RGB(self, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH

        count = 0
        mean_r = []
        mean_g = []
        mean_b = []
        std_r = []
        std_g = []
        std_b = []

        for data_filename in data_filenames:
            data_file = "{}/{}".format(root_path, data_filename)

            f = open(data_file, "r")
            for line in f:
                items = line.split(" ")
                patch_file = "{}/{}".format(root_path, items[0])
                img = io.imread(patch_file, as_gray=False)

                # lab_img = color.rgb2lab(img)

                # RGB三通道分离
                rgb_r = img[:, :, 0] / 255.0
                rgb_g = img[:, :, 1] / 255.0
                rgb_b = img[:, :, 2] / 255.0

                # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
                std_r.append(np.std(rgb_r))
                std_g.append(np.std(rgb_g))
                std_b.append(np.std(rgb_b))

                mean_r.append(np.mean(rgb_r))
                mean_g.append(np.mean(rgb_g))
                mean_b.append(np.mean(rgb_b))

                if (0 == count%1000):
                    print("{} calculate mean and std >>> {}".format(time.asctime( time.localtime()), count))
                count += 1

            f.close()

        avg_mean_r = np.mean(mean_r)
        avg_mean_g = np.mean(mean_g)
        avg_mean_b = np.mean(mean_b)
        avg_std_r = np.mean(std_r)
        avg_std_g = np.mean(std_g)
        avg_std_b = np.mean(std_b)

        return avg_mean_r, avg_mean_g, avg_mean_b, avg_std_r, avg_std_g, avg_std_b

    @staticmethod
    def normalize_mean(src_img, *args):
        pass
