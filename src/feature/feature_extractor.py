#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-16'

"""
import time
from skimage import feature, color, util, exposure
import numpy as np
from skimage import io
from preparation.normalization import ImageNormalization


class FeatureExtractor(object):
    def __init__(self, params):
        self._params = params
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
        elif code == "best":
            return self.extract_best_feature(src_img)
        elif code == "most":
            return self.extract_most_feature(src_img)

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
        # image = exposure.rescale_intensity(color.rgb2gray(src_img), out_range=(0, 255))
        # 计算灰度共生矩阵
        # glcm = feature.greycomatrix(image, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True) # 效果不如下面的好
        glcm = feature.greycomatrix(image, [3], [0], 256, symmetric=True, normed=True)
        feaprops_names = ('contrast', 'dissimilarity',  'homogeneity','ASM','energy','correlation')
        # 得到不同统计量
        for fname in feaprops_names:
            fes = feature.greycoprops(glcm, fname)
            textural_feature.extend(np.array(fes).flatten())

        return  textural_feature

    def extract_best_feature(self, src_img):
        '''
        提取图像的GLCM特征 和 平均亮度
        :param src_img: 输入图像
        :return: GLCM特征
        '''
        # 存储单张图片的glcm特征
        textural_feature = []
        # 以灰度模式读取图片
        # image =  util.img_as_ubyte(color.rgb2gray(src_img))
        # image = np.array(255 * color.rgb2gray(src_img)).astype(np.uint8)
        image = np.array(exposure.rescale_intensity(color.rgb2gray(src_img), out_range=(0, 255))).astype(np.uint8)
        # 计算灰度共生矩阵
        # glcm = feature.greycomatrix(image, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True) # 效果不如下面的好
        glcm = feature.greycomatrix(image, [1,3,5], [0], 256, symmetric=True, normed=True)
        # 得到不同统计量
        fes = feature.greycoprops(glcm, 'dissimilarity')
        textural_feature.extend(np.array(fes).flatten())

        meanValue = np.mean(image)
        textural_feature.append(meanValue)
        return  textural_feature

    def extract_most_feature(self, src_img):
        '''
        提取图像的GLCM特征 和 平均亮度
        :param src_img: 输入图像
        :return: GLCM特征
        '''
        # 存储单张图片的glcm特征
        textural_feature = []
        # 以灰度模式读取图片
        image = np.array(exposure.rescale_intensity(color.rgb2gray(src_img), out_range=(0, 255))).astype(np.uint8)
        # 计算灰度共生矩阵
        glcm = feature.greycomatrix(image, [1,3,5], [0], 256, symmetric=True, normed=True)
        # 得到不同统计量
        # fes = feature.greycoprops(glcm, 'dissimilarity')
        # textural_feature.extend(np.array(fes).flatten())
        feaprops_names = ('contrast', 'dissimilarity',  'homogeneity','ASM','energy','correlation')
        # 得到不同统计量
        for fname in feaprops_names:
            fes = feature.greycoprops(glcm, fname)
            textural_feature.extend(np.array(fes).flatten())

        meanValue = np.mean(image)
        textural_feature.append(meanValue)
        return  textural_feature

    def extract_features_by_file_list(self, data_filename, features_name = "best"):
        '''
        从指定文件列表中，读入图像文件，并计算特征，和分类Tag
        :param data_filename: 图像文件的列表，前项是文件名，后项是tag
        :return: 特征向量集合，tag集合
        '''
        root_path = self._params.PATCHS_ROOT_PATH
        data_file = "{}/{}".format(root_path, data_filename)

        features = []
        tags = []
        count = 0

        f = open(data_file, "r")
        for line in f:
            items = line.split(" ")
            patch_file = "{}/{}".format(root_path, items[0])
            img = io.imread(patch_file, as_gray=False)
            tag = int(items[1])

            normal_img = ImageNormalization.normalize_mean(img)
            fvector = self.extract_feature(normal_img, features_name)

            features.append(fvector)
            tags.append(tag)

            if (0 == count % 200):
                print("{} extract feature >>> {}".format(time.asctime(time.localtime()), count))
            count += 1

        f.close()
        return features, tags

