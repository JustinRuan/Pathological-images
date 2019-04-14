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
from core.util import read_csv_file
from skimage.io import imread
from core import Block

class ImageAugmentation(object):

    def __init__(self, **kwarg):
        # 实验2用：l_range = (0.9, 1.1), a_range = (0.95, 1.05), b_range = (0.95, 1.05), constant_range = (-10, 10)
        # 实验1用：l_range = (0.99, 1.01), a_range = (0.99, 1.01), b_range = (0.99, 1.01), constant_range = (-5, 5)
        l_range = kwarg["l_range"]
        a_range = kwarg["a_range"]
        b_range = kwarg["b_range"]
        constant_range = kwarg["constant_range"]

        K = 4
        self.l_candidates = np.arange(l_range[0], 1.01 * l_range[1], (l_range[1] - l_range[0]) / K)
        self.a_candidates = np.arange(a_range[0], 1.01 * a_range[1], (a_range[1] - a_range[0]) / K)
        self.b_candidates = np.arange(b_range[0], 1.01 * b_range[1], (b_range[1] - b_range[0]) / K)
        self.constant_candidates = np.arange(constant_range[0], 1.01 * constant_range[1],
                                             (constant_range[1] - constant_range[0]) / K)
        self.K = K

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
        K = self.K

        f1_l = self.l_candidates[random.randint(0, K)]
        f1_c = self.constant_candidates[random.randint(0, K)]
        labO_l[tag] = f1_l * labO_l[tag] +  f1_c # Adjust Luminance

        f2_a = self.a_candidates[random.randint(0, K)]
        f2_c = self.constant_candidates[random.randint(0, K)]
        labO_a[tag] = f2_a * labO_a[tag] + f2_c  # Adjust color

        f3_b = self.b_candidates[random.randint(0, K)]
        f3_c = self.constant_candidates[random.randint(0, K)]
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

    def augment_dataset(self, params, source_samples, tagrget_dir, max_count = None):
        patch_root = params.PATCHS_ROOT_PATH[source_samples[0]]
        sample_filename = source_samples[1]
        train_list = "{}/{}".format(patch_root, sample_filename)

        Xtrain, Ytrain = read_csv_file(patch_root, train_list)
        if max_count is not None:
            Xtrain = Xtrain[:max_count]
            Ytrain = Ytrain[:max_count]

        target_cancer_path = "{}/{}_cancer".format(patch_root, tagrget_dir)
        target_normal_path = "{}/{}_noraml".format(patch_root, tagrget_dir)

        if (not os.path.exists(target_cancer_path)):
            os.makedirs(target_cancer_path)
        if (not os.path.exists(target_normal_path)):
            os.makedirs(target_normal_path)

        for x, y in zip(Xtrain, Ytrain):
            block = Block()
            block.load_img(x)
            img = block.get_img()

            aug_img = self.augment_images(img) * 255
            block.set_img(aug_img)
            block.opcode = 9

            if y == 0:
                block.save_img(target_normal_path)
            else:
                block.save_img(target_cancer_path)



