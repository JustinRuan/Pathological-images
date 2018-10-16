#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-16'

"""
from skimage import feature, color, util
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        # self._params = params
        return

    def extract_feature(self, src_img, code):
        '''
        对输入图像进行特征提取
        :param src_img: 输入图像
        :param code: 特征的编号
        :return: 特征向量
        '''
        if code == "glcm":
            return self.extract_glcm_feature(src_img)
        elif code == "blp":
            return

    def extract_glcm_feature(self, src_img):
        '''
        提取图像的GLCM特征
        :param src_img: 输入图像
        :return: GLCM特征
        '''
        # 存储单张图片的glcm特征
        textural_feature = []
        # 以灰度模式读取图片
        image =  util.img_as_ubyte(color.rgb2gray(src_img))
        # image = util.img_as_ubyte(src_img)
        # 计算灰度共生矩阵
        # glcm = feature.greycomatrix(image, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
        glcm = feature.greycomatrix(image, [5], [0], 256, symmetric=True,
                                    normed=True)
        feaprops_names = ('contrast', 'dissimilarity',  'homogeneity','ASM','energy','correlation')
        # 得到不同统计量
        for fname in feaprops_names:
            fes = feature.greycoprops(glcm, fname)
            textural_feature.extend(np.array(fes).flatten())

        return  textural_feature