#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-24'

"""
from skimage import feature, color, util

class FeatureExtractor(object):
    def __init__(self):
        # self._params = params
        return

    def extract_feature(self, src_img, code):
        if code == "glcm":
            return self.extract_glcm_feature(src_img)
        elif code == "blp":
            return

    def extract_glcm_feature(self, src_img):
        # 存储单张图片的glcm特征
        textural_feature = []
        # 以灰度模式读取图片
        image =  util.img_as_ubyte(color.rgb2gray(src_img))
        # image = util.img_as_ubyte(src_img)
        # 计算灰度共生矩阵
        glcm = feature.greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
        # 得到不同统计量
        textural_feature.append(feature.greycoprops(glcm, 'contrast')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'dissimilarity')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'homogeneity')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'ASM')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'energy')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'correlation')[0, 0])

        return  textural_feature