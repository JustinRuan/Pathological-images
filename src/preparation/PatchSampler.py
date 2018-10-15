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
        :param seeds_name: 当前中心点集合的编码或代号（C，N，E）
        :return: 某个文件夹中的图块文件（cancer, normal, edge三个之一）
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

    def extract_patches_RZone(self, sourceCone, scale):
        '''
        从粗标区提取图块，并根据 SVM的分类结果存到对应目录。

        :param sourceCone: 切片
        :param scale: 提取图块所用分辨率
        :return: 五种文件夹的图块文件
        '''
        if (scale != self.extract_scale):
            print("\a", "scale error!")
            return

        Root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(self.extract_scale * 100).astype(np.int)

        pathCancer = "{}/S{}_{}".format(Root_path,intScale, "cancerR")
        pathNormal = "{}/S{}_{}".format(Root_path,intScale, "normalR")
        pathUnsure = "{}/S{}_{}".format(Root_path,intScale, "unsure")
        pathFalseCancer = "{}/S{}_{}".format(Root_path,intScale, "falseCancerR")
        pathFalseNormal = "{}/S{}_{}".format(Root_path, intScale, "falseNormalR")

        if (not os.path.exists(pathCancer)):
            os.makedirs(pathCancer)

        if (not os.path.exists(pathNormal)):
            os.makedirs(pathNormal)

        if (not os.path.exists(pathUnsure)):
            os.makedirs(pathUnsure)

        if (not os.path.exists(pathFalseCancer)):
            os.makedirs(pathFalseCancer)

        if (not os.path.exists(pathFalseNormal)):
            os.makedirs(pathFalseNormal)

        patch_size = self._params.PATCH_SIZE_HIGH

        pf = PatchFeature.PatchFeature(self._params)
        classifier = pf.load_svm_model()
        extractor = FeatureExtractor.FeatureExtractor()

        # 对于癌变粗标区进行处理
        lenTR = len(self.seeds_TR)
        for idx, (x, y) in enumerate(self.seeds_TR):
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            img = block.get_img()
            feature = extractor.extract_glcm_feature(img)
            predicted = classifier.predict_proba([feature])
            tag = self.detect_patch_byProb(predicted[0])
            if (tag == 1):
                block.save_img(pathCancer)
            elif (tag == 0):
                print("{} / {} Find Normal patch in TR Zone, {}".format(idx, lenTR, block.encoding()))
                block.save_img(pathFalseCancer)
            else:
                print("{} / {} Find Unsure patch in TR Zone, {} {}".format(idx, lenTR, block.encoding(), predicted[0]))
                block.save_img(pathUnsure)

            if (idx % 1000 == 0):
                print("TR -> Processing: {} / {}".format(idx, lenTR))

        # 对于正常粗标区进行处理
        lenNR = len(self.seeds_NR)
        for idx, (x, y) in enumerate(self.seeds_NR):
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            img = block.get_img()
            feature = extractor.extract_glcm_feature(img)
            predicted = classifier.predict_proba([feature])
            tag = self.detect_patch_byProb(predicted[0])
            if (tag == 1):
                print("{} / {} Find Cancer patch in NR Zone, {}".format(idx, lenNR, block.encoding()))
                block.save_img(pathFalseNormal)
            elif (tag == 0):
                block.save_img(pathNormal)
            else:
                print("{} / {} Find Unsure patch in NR Zone, {} {}".format(idx,  lenNR, block.encoding(), predicted[0]))
                block.save_img(pathUnsure)

            if (idx % 1000 == 0):
                print("NR -> Processing: {} / {}".format(idx, lenNR))

        return

    def detect_patch_byProb(self, probs):
        '''
        根据 给定阈值判定图块的类型
        :param probs: 对应分类（0：正常，1：癌）的概率值
        :return: -1，0，1
        '''
        normal_prob = probs[0]
        cancer_prob = probs[1]
        if (normal_prob > 0.90): # 正常
            return 0
        elif (cancer_prob > 0.90): # 癌变
            return 1
        else:   # 不能确定
            return -1