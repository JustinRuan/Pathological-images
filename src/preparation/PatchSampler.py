#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import numpy as np
from skimage import color, morphology
from skimage.morphology import square
import os
# from preparation import PatchFeature
from feature import feature_extractor
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

    def extract_patches(self, sourceCone, extract_scale, patch_size, seeds, seeds_name):
        '''
        从给定的中心点集合提取图块，并存到对应目录
        :param sourceCone: 切片
        :param extract_scale: 提取图块所用分辨率
        :param patch_size: 切片的边长
        :param seeds: 中心点集合
        :param seeds_name: 当前中心点集合的编码或代号（cancer, edge, noraml, 三个之一）
        :return: 某个文件夹中的图块文件
        '''
        Root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(extract_scale * 100).astype(np.int)

        pathPatch = "{}/S{}_{}_{}".format(Root_path, intScale, patch_size, seeds_name)

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

        C_mask = mask["C"]
        EI_mask = mask["EI"]
        EO_mask = mask["EO"]

        c_seeds = get_seeds(C_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=-8)
        ei_seeds = get_seeds(EI_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)
        eo_seeds = get_seeds(EO_mask, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)

        return c_seeds, ei_seeds, eo_seeds

    def get_multi_scale_seeds(self,extract_scale_list, seeds, seeds_scale):
        seeds_dict = {}
        seeds_dict[seeds_scale] = seeds

        for extract_scale in extract_scale_list:
            seeds_dict[extract_scale] = transform_coordinate(0, 0, 1.25, seeds_scale, extract_scale, seeds)

        return seeds_dict

    def detect_normal_patches_with_scale(self, sourceCone, extract_scale, patch_size, sampling_interval):
        low_scale = self._params.GLOBAL_SCALE
        eff_region = sourceCone.get_effective_zone(low_scale)
        n_seeds = get_seeds(eff_region, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=0)

        return n_seeds

    def extract_patches_multi_scale(self, sourceCone, seeds_dict, patch_size, seeds_name):
        for scale, seeds in seeds_dict.items():
            self.extract_patches(sourceCone, scale, patch_size, seeds, seeds_name)

