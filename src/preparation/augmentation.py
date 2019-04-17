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
import random
from core.util import read_csv_file, get_project_root
from core import Block
import random

class AbstractAugmentation(object):
    def augment_images(self, src_img):
        raise NotImplementedError

    def process(self, src_img):
        return self.augment_images(src_img)

    def augment_dataset(self, params, source_samples, tagrget_dir, range = None):
        patch_root = params.PATCHS_ROOT_PATH[source_samples[0]]
        sample_filename = source_samples[1]
        train_list = "{}/{}".format(patch_root, sample_filename)

        Xtrain, Ytrain = read_csv_file(patch_root, train_list)
        if range is not None:
            Xtrain = Xtrain[range[0]:range[1]]
            Ytrain = Ytrain[range[0]:range[1]]

        target_cancer_path = "{}/{}_cancer".format(patch_root, tagrget_dir)
        target_normal_path = "{}/{}_noraml".format(patch_root, tagrget_dir)

        if (not os.path.exists(target_cancer_path)):
            os.makedirs(target_cancer_path)
        if (not os.path.exists(target_normal_path)):
            os.makedirs(target_normal_path)

        for K, (x, y) in enumerate(zip(Xtrain, Ytrain)):
            block = Block()
            block.load_img(x)
            img = block.get_img()

            aug_img = self.augment_images(img) * 255
            block.set_img(aug_img)
            block.opcode = self.opcode

            if y == 0:
                block.save_img(target_normal_path)
            else:
                block.save_img(target_cancer_path)

            if (0 == K % 1000):
                print("{} augmenting >>> {}".format(time.asctime(time.localtime()), K))


class ImageAugmentation(AbstractAugmentation):

    def __init__(self, **kwarg):
        # 实验2用：l_range = (0.9, 1.1), a_range = (0.95, 1.05), b_range = (0.95, 1.05), constant_range = (-10, 10)
        # 实验1用：l_range = (0.99, 1.01), a_range = (0.99, 1.01), b_range = (0.99, 1.01), constant_range = (-5, 5)
        l_range = kwarg["l_range"]
        a_range = kwarg["a_range"]
        b_range = kwarg["b_range"]
        constant_range = kwarg["constant_range"]

        K = 10
        self.l_candidates = np.arange(l_range[0], 1.01 * l_range[1], (l_range[1] - l_range[0]) / K)
        self.a_candidates = np.arange(a_range[0], 1.01 * a_range[1], (a_range[1] - a_range[0]) / K)
        self.b_candidates = np.arange(b_range[0], 1.01 * b_range[1], (b_range[1] - b_range[0]) / K)
        self.constant_candidates = np.arange(constant_range[0], 1.01 * constant_range[1],
                                             (constant_range[1] - constant_range[0]) / K)
        self.K = K
        self.opcode = 9

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


class HistAugmentation(AbstractAugmentation):

    def __init__(self, **kwarg):
        target_path = "{}/data/{}".format(get_project_root(), kwarg["hist_target"])
        hist_target = np.load(target_path).item()
        self.hist_target = hist_target
        self.opcode = 10

        if kwarg["hist_source"] is not None:
            print("reading histogram file ...")
            source_path = "{}/data/{}".format(get_project_root(), kwarg["hist_source"])
            print("reading histogram file: ", source_path)
            hist_source = np.load(source_path).item()

            LUT = []
            LUT.append(self._estimate_cumulative_cdf(hist_source["L"], hist_target["L"], start=0, end=100))
            LUT.append(self._estimate_cumulative_cdf(hist_source["A"], hist_target["A"], start=-128, end=127))
            LUT.append(self._estimate_cumulative_cdf(hist_source["B"], hist_target["B"], start=-128, end=127))
            self.LUT = LUT
            self.hist_source = hist_source
            self.hist_target = hist_target
        else:
            # 将使用Prepare过程进行初始化
            self.LUT = None
            self.hist_source = None


    def augment_images(self, src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        lab0_l = np.array(lab_img[:, :, 0]).astype(np.int)
        lab0_a = np.array(lab_img[:, :, 1]).astype(np.int)
        lab0_b = np.array(lab_img[:, :, 2]).astype(np.int)

        LUT_L = self.LUT[0]
        lab1_l = LUT_L[lab0_l]

        LUT_A = self.LUT[1]
        lab1_a = LUT_A[128 + lab0_a]

        LUT_B = self.LUT[2]
        lab1_b = LUT_B[128 + lab0_b]

        labO = np.dstack([lab1_l, lab1_a, lab1_b])
        # LAB to RGB变换, 会除以255
        rgb_image = color.lab2rgb(labO)

        return rgb_image

    def _estimate_cumulative_cdf(self, source, template, start, end):
        src_values, src_counts = source
        tmpl_values, tmpl_counts = template

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / np.sum(src_counts)
        tmpl_quantiles = np.cumsum(tmpl_counts) / np.sum(tmpl_counts)

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

        if src_values[0] > start:
            src_values = np.insert(src_values, 0, start)
            interp_a_values = np.insert(interp_a_values, 0, start)
        if src_values[-1] < end:
            src_values = np.append(src_values, end)
            interp_a_values = np.append(interp_a_values, end)

        new_source = np.arange(start, end + 1)
        interp_b_values = np.interp(new_source, src_values, interp_a_values)
        # result = dict(zip(new_source, np.rint(interp_b_values))) # for debug
        # return result
        return np.rint(interp_b_values)

class RndAugmentation(AbstractAugmentation):

    def __init__(self, **kwarg):
        self.opcode = 11

    def augment_images(self, src_img):
        rnd_index = [0, 1, 2]
        random.shuffle(rnd_index)

        # RGB三通道分离
        if random.random() > 0.8:
            rgb_r = 255 - src_img[:, :, rnd_index[0]]
            rgb_g = 255 - src_img[:, :, rnd_index[1]]
            rgb_b = 255 - src_img[:, :, rnd_index[2]]
        else:
            rgb_r = src_img[:, :, rnd_index[0]]
            rgb_g = src_img[:, :, rnd_index[1]]
            rgb_b = src_img[:, :, rnd_index[2]]

        rgb = np.dstack([rgb_r, rgb_g, rgb_b])

        return rgb