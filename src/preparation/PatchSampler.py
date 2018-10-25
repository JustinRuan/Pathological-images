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
from preparation import PatchFeature
from feature import FeatureExtractor
from core.util import get_seeds

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
        # C_mask, N_mask, E_mask = sourceCone.create_mask_image(lowScale, edgeWidth)
        # mask1 = sourceCone.get_effective_zone(lowScale)
        # N_mask = N_mask & mask1

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
        :param seeds_name: 当前中心点集合的编码或代号（cancer, stroma, edge, lymph四个之一）
        :return: 某个文件夹中的图块文件
        '''
        Root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(extract_scale * 100).astype(np.int)

        pathPatch = "{}/S{}_{}_{}".format(Root_path, intScale, patch_size, seeds_name)

        if (not os.path.exists(pathPatch)):
            os.makedirs(pathPatch)

        for (x, y) in seeds:
            block = sourceCone.get_image_block(extract_scale, x, y, patch_size, patch_size)
            block.save_img(pathPatch)

        return
