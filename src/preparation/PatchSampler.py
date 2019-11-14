#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import os

import numpy as np

# from preparation import PatchFeature
from core.util import get_seeds, transform_coordinate


class PatchSampler(object):
    def __init__(self, params):
        self._params= params

    def generate_seeds4_high(self, mask,  extract_scale, patch_size, patch_spacing):
        '''
        根据切片文件标注区域对应的Mask图像，在高分辨率上提取图块种子点（中心）的坐标
        :param mask: 使用lowScale生成mask图像，即GLOBAL_SCALE倍镜下的Mask
        :param extract_scale: 提取图块所用分辨率
        :param patch_size: 提取图块的大小，间距是边长的一半
        :param patch_spacing: 图块之间的间距
        :return: 返回种子点的集合的大小
        '''

        lowScale = self._params.GLOBAL_SCALE
        new_seeds = get_seeds(mask, lowScale, extract_scale, patch_size, patch_spacing)

        return new_seeds

    def extract_patches(self, sourceCone, extract_scale, patch_size, seeds, seeds_name, samples_dir):
        '''
        从给定的中心点集合提取图块，并存到对应目录
        :param sourceCone: 切片
        :param extract_scale: 提取图块所用分辨率
        :param patch_size: 切片的边长
        :param seeds: 中心点集合
        :param seeds_name: 当前中心点集合的编码或代号（cancer, edge, noraml, 三个之一）
        :return: 某个文件夹中的图块文件
        '''
        Root_path = self._params.PATCHS_ROOT_PATH[samples_dir]
        intScale = np.rint(extract_scale * 100).astype(np.int)

        pathPatch = "{}/S{}_{}_{}/{}".format(Root_path, intScale, patch_size, seeds_name, sourceCone.slice_id)

        if not os.path.exists(pathPatch):
            os.makedirs(pathPatch)

        for (x, y) in seeds:
            block = sourceCone.get_image_block(extract_scale, x, y, patch_size, patch_size)
            block.save_img(pathPatch)

        return

    def detect_cancer_patches_with_scale(self, sourceCone, extract_scale, patch_size, sampling_interval, edge_width):
        low_scale = self._params.GLOBAL_SCALE
        # edge_width = 4  # (256 / (40 / 1.25) = 8, 256的图块在1.25倍镜下边长为8)

        mask = sourceCone.create_mask_image(low_scale, edge_width)
        roi = sourceCone.get_effective_zone(low_scale)

        C_mask = mask["C"]
        N_mask = mask["N"] & roi
        EI_mask = mask["EI"]
        EO_mask = mask["EO"]

        sum_C = np.sum(C_mask)
        sum_N = np.sum(N_mask)
        ratio = np.sqrt(sum_N / sum_C)
        ratio = min(5, ratio)

        if ratio > 1:
            normal_sampling_interval = np.rint(ratio * sampling_interval)
        else:
            normal_sampling_interval = sampling_interval

        len_c = 0
        len_n = 0
        iter_count = 0
        while ((len_n < 2000) or (len_c < 2000)) and iter_count < 6:
            print(sourceCone.slice_id, "Cancer sampling interval:", sampling_interval, "normal sampling interval:",
                  normal_sampling_interval)

            c_seeds = get_seeds(C_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=-2)
            ei_seeds = get_seeds(EI_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)
            eo_seeds = get_seeds(EO_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)
            n_seeds = get_seeds(N_mask, low_scale, extract_scale, patch_size, spacingHigh=normal_sampling_interval, margin=-2)

            c_seeds = set(c_seeds)
            ei_seeds = set(ei_seeds)
            eo_seeds = set(eo_seeds)
            c_and_ei = c_seeds & ei_seeds
            c_and_eo = c_seeds & eo_seeds
            eo_and_ei = eo_seeds & ei_seeds

            result_c = list(c_seeds - c_and_ei - c_and_eo)
            result_ei = list(ei_seeds - eo_and_ei)
            result_eo = list(eo_seeds - eo_and_ei)
            result_n = list(set(n_seeds) - c_seeds - ei_seeds - eo_seeds)

            len_c = len(result_c) + len(result_ei)
            len_n = len(result_n) + len(result_eo)
            if len_c < 2000:
                sampling_interval = np.rint(0.9 * sampling_interval).astype(np.int)
            if len_n < 2000:
                normal_sampling_interval = np.rint(0.9 * normal_sampling_interval).astype(np.int)
            iter_count += 1

        return result_c, result_ei, result_eo, result_n

    def get_multi_scale_seeds(self,extract_scale_list, seeds, seeds_scale):
        seeds_dict = {}
        seeds_dict[seeds_scale] = seeds

        if extract_scale_list is not None and len(extract_scale_list) > 0 :
            for extract_scale in extract_scale_list:
                seeds_dict[extract_scale] = transform_coordinate(0, 0, 1.25, seeds_scale, extract_scale, seeds)

        return seeds_dict

    def detect_normal_patches_with_scale(self, sourceCone, extract_scale, patch_size, sampling_interval, c_mask = None):
        low_scale = self._params.GLOBAL_SCALE
        eff_region = sourceCone.get_effective_zone(low_scale)

        if c_mask is not None:
            eff_region = np.bitwise_and(eff_region, np.bitwise_not(c_mask))

        len_n = 0
        iter_count = 0
        while len_n < 2000 and iter_count < 6:
            n_seeds = get_seeds(eff_region, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)
            len_n = len(n_seeds)
            if len_n < 2000:
                sampling_interval = np.rint(0.9 * sampling_interval).astype(np.int)
            iter_count += 1

        return n_seeds

    def extract_patches_multi_scale(self, sourceCone, seeds_dict, patch_size, seeds_name, samples_dir):
        for scale, seeds in seeds_dict.items():
            self.extract_patches(sourceCone, scale, patch_size, seeds, seeds_name, samples_dir)


