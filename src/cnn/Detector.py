#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""

from core import *
import caffe
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io, util
from sklearn.svm import NuSVC, SVC
from sklearn.externals import joblib
from skimage import segmentation
from skimage.draw import rectangle # 需要skimage 0.14及以上版本
from core.util import get_seeds
from cnn import transfer_cnn
import random


class Detector(object):

    def __init__(self, config_file, slice_file, ano_filename, id_string):
        '''
        初始化 检测器
        :param config_file: 配置文件
        :param slice_file: 切片文件
        :param ano_filename: 对应的标注文件
        :param id_string: 切片编号
        '''
        self._params = Params.Params()
        self._params.load_config_file(config_file)

        self.imgCone = ImageCone.ImageCone(self._params)

        # 读取数字全扫描切片图像
        tag = self.imgCone.open_slide(slice_file, ano_filename, id_string)

        if tag:
            w, h = self.imgCone.get_image_width_height_byScale(self._params.GLOBAL_SCALE)
            self.ImageWidth = w
            self.ImageHeight = h
            self.ROI = np.zeros((h, w), dtype=np.bool)
        return

    def get_ROI_img(self, x1, y1, x2, y2, coordinate_scale, img_scale):
        '''
        得到ROI区域的图像
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :param coordinate_scale: 上面四个坐标所在的倍镜
        :param img_scale:生成ROI图像所用的倍镜
        :return: ROI图像
        '''
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * img_scale / coordinate_scale).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1
        block = self.imgCone.get_image_block(img_scale, xx1, yy1, w, h)
        return block.get_img()

    def set_ROI(self, x1, y1, x2, y2, scale):
        '''
        设置矩形的ROI区域
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :param scale: 上面四个坐标所在的倍镜
        :return:
        '''
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2 ])* GLOBAL_SCALE / scale).astype(np.int)
        rr, cc = rectangle((yy1, xx1), end=(yy2, xx2))
        self.ROI[rr, cc] = 1
        self.ROI_width = xx2 - xx1
        self.ROI_height = yy2 - yy1
        return

    def reset_ROI(self):
        '''
        清空ROI中的标记
        :return:
        '''
        self.ROI =  np.zeros((self.ImageHeight, self.ImageWidth), dtype=np.bool)
        return

    def get_points_ROI(self, highScale, patch_size_high, spacingHigh):
        '''
        得到ROI区中的点在高分辨率下的坐标
        :param highScale: 高分辨率对应的倍镜数
        :param patch_size_high: 高分辨率下的图块大小
        :param spacingHigh: 高分辨率下的图块之间的距离
        :return: （x，y）种子点的集合
        '''
        return get_seeds(self.ROI, self._params.GLOBAL_SCALE, highScale,patch_size_high, spacingHigh, margin=8)

    # highScale, spacingHigh, block_size 都是在高分辨率下的值
    def get_imagelist_ROI(self, highScale, spacingHigh, block_size):
        '''
        得到ROI区中在高分辨率下的图块集合
        :param highScale: 高分辨率对应的倍镜数
        :param spacingHigh: 高分辨率下的图块之间的距离
        :param block_size: 高分辨率下的图块大小
        :return: （x，y）种子点的集合， 图块集合
        '''
        seeds = self.get_points_ROI(highScale, block_size, spacingHigh)

        images = []
        for (x, y) in seeds:
            block = self.imgCone.get_image_block(highScale, x, y, block_size, block_size)
            images.append(block.get_img())
        return seeds, images

    def detect_ROI(self, scale, block_size):
        '''
        对设定的ROI区域进行检测，判定图块的类型：0 正常，1 癌变
        :param scale: 提取图块所用的倍镜数
        :param block_size: 图块大小
        :return: 返回ROI中的种子点的坐标（左上角，高倍镜下），预测的分类结果
        '''
        spacing = block_size / 2

        seeds, images = self.get_imagelist_ROI(scale, spacing, block_size)

        tc = transfer_cnn.transfer_cnn(self._params, "googlenet", "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt")
        tc.start_caffe()
        glcm_features, cnn_features = tc.extract_cnn_feature(images)
        # print(np.array(cnn_features)[:, 0:3])

        features = tc.comobine_features(glcm_features, cnn_features)
        classifier = tc.load_svm_model()
        predicted_tags = classifier.predict(features)
        return seeds, predicted_tags

    def draw_result(self, seeds, seeds_scale, block_size_high, tags, x1, y1, xy_scale):
        '''
        根据图块的分类结果，按图块覆盖的区域进行打分，画出对应的区域
        :param seeds: 提取的ROI中的种子点的坐标（左上角，高倍镜下）
        :param seeds_scale: 种子点坐标的倍镜
        :param block_size_high: 高倍镜（高分辨率）下，图块的大小
        :param tags: 预测的结果，0 正常，1 癌变
        :param x1: ROI区域在全局切片中左上角x
        :param y1: ROI区域在全局切片中左上角y
        :param xy_scale:生成ROI区域所使用的倍镜数
        :return: 打分后图像
        '''
        result = np.zeros((self.ROI_height, self.ROI_width), dtype=np.int)
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        # 所有坐标都统一到 GLOBAL_SCALE 下
        xx1 = int(x1 * GLOBAL_SCALE / xy_scale)
        yy1 = int(y1 * GLOBAL_SCALE / xy_scale)
        block_size_low = int(block_size_high * GLOBAL_SCALE / seeds_scale)

        for ((x, y), tag) in zip(seeds, tags):
            xx = int(x * GLOBAL_SCALE / seeds_scale - xx1)
            yy = int(y * GLOBAL_SCALE / seeds_scale - yy1)
            # print(xx, yy)
            rr, cc = rectangle((yy, xx), extent=(block_size_low, block_size_low))

            if tag > -1 :
                select_y = (rr >= 0) & (rr < self.ROI_height)
                select_x = (cc >= 0) & (cc < self.ROI_width)
                select = select_x & select_y
                result[rr[select], cc[select]] = result[rr[select], cc[select]] + tag

        return result

    def segment_image(self, src_img, count):
        '''
        对图像进行聚类
        :param src_img: 输入的全局图像
        :param count: 分割区域的最大数量
        :return: 聚类后的Mask矩阵
        '''
        segments = segmentation.slic(src_img, n_segments=count,compactness=5, sigma=3)
        return segments

    # def detect_ROI_regions(self, x1, y1, x2, y2, coordinate_scale, regions_count, extract_scale, block_size):
    #     '''
    #     在聚类的基础上，对每个区域进行抽样检测，用检测得到的平均值对区域进行打分
    #     :param x1: ROI区域的左上角x
    #     :param y1: ROI区域的左上角y
    #     :param x2: ROI区域的右下角x
    #     :param y2: ROI区域的右下角y
    #     :param coordinate_scale: 生成ROI区域所用倍镜
    #     :param regions_count: 分割出区域的最大数量
    #     :param extract_scale: 提取图块所使用的倍镜
    #     :param block_size: 提取图块的大小（在extract_scale下）
    #     :return: 聚类区域，打分图像
    #     '''
    #     ###########################################################
    #     tc = transfer_cnn.transfer_cnn(self._params, "googlenet", "bvlc_googlenet.caffemodel",
    #                                    "deploy_GoogLeNet.prototxt")
    #     tc.start_caffe()
    #
    #     classifier = tc.load_svm_model()
    #     ###########################################################
    #     result = np.zeros((self.ROI_height, self.ROI_width), dtype=np.float)
    #
    #     spacing = block_size / 2
    #     GLOBAL_SCALE = self._params.GLOBAL_SCALE
    #
    #     roi_img = self.get_ROI_img(x1, y1, x2, y2, coordinate_scale, GLOBAL_SCALE)
    #     regions = self.segment_image(roi_img, regions_count)
    #
    #     # 计算在extract时的左上解坐标
    #     xx1 = int(x1 * extract_scale / coordinate_scale)
    #     yy1 = int(y1 * extract_scale / coordinate_scale)
    #
    #     regions_count = np.max(regions)
    #     thresh = 30
    #     for index in range(regions_count + 1):
    #         self.ROI = (regions == index)
    #         seeds = self.get_points_ROI(extract_scale, block_size, spacing)
    #         list_seeds = list(seeds)
    #         if len(list_seeds) > thresh:
    #             # 抽取
    #             random.shuffle(list_seeds)
    #             list_seeds = list_seeds[:thresh]
    #
    #         images = []
    #         for (x, y) in list_seeds:
    #             block = self.imgCone.get_image_block(extract_scale, x + xx1 , y + yy1, block_size, block_size)
    #             images.append(block.get_img())
    #
    #         glcm_features, cnn_features = tc.extract_cnn_feature(images)
    #         features = tc.comobine_features(glcm_features, cnn_features)
    #
    #         predicted_tags = classifier.predict(features)
    #         mean_tags = np.mean(predicted_tags)
    #         print(index,'/',regions_count, "->", mean_tags, predicted_tags)
    #         result[self.ROI] = mean_tags
    #
    #     return regions, result

    def detect_ROI_regions(self, x1, y1, x2, y2, coordinate_scale, regions_count, extract_scale, block_size):
        '''
        在聚类的基础上，对每个区域进行抽样检测，用检测得到的平均值对区域进行打分
        :param x1: ROI区域的左上角x
        :param y1: ROI区域的左上角y
        :param x2: ROI区域的右下角x
        :param y2: ROI区域的右下角y
        :param coordinate_scale: 生成ROI区域所用倍镜
        :param regions_count: 分割出区域的最大数量
        :param extract_scale: 提取图块所使用的倍镜
        :param block_size: 提取图块的大小（在extract_scale下）
        :return: 聚类区域，打分图像
        '''
        ###########################################################
        tc = transfer_cnn.transfer_cnn(self._params, "googlenet", "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt")
        tc.start_caffe()

        classifier = tc.load_svm_model()
        ###########################################################
        result = np.zeros((self.ROI_height, self.ROI_width), dtype=np.float)

        spacing = block_size / 2
        GLOBAL_SCALE = self._params.GLOBAL_SCALE

        roi_img = self.get_ROI_img(x1, y1, x2, y2, coordinate_scale, GLOBAL_SCALE)
        regions = self.segment_image(roi_img, regions_count)

        # 计算在extract时的左上解坐标
        xx1 = int(x1 * extract_scale / coordinate_scale)
        yy1 = int(y1 * extract_scale / coordinate_scale)

        regions_count = np.max(regions)
        thresh = 30
        for index in range(regions_count + 1):
            self.ROI = (regions == index)
            seeds = self.get_points_ROI(extract_scale, block_size, spacing)
            list_seeds = list(seeds)
            if len(list_seeds) > thresh:
                # 抽取
                random.shuffle(list_seeds)
                list_seeds = list_seeds[:thresh]

            # images = []
            # for (x, y) in list_seeds:
            #     block = self.imgCone.get_image_block(extract_scale, x + xx1 , y + yy1, block_size, block_size)
            #     images.append(block.get_img())
            seeds_array = np.array(list_seeds)
            img_itor = self.imgCone.get_image_blocks_itor(extract_scale, seeds_array[:,0] + xx1 , seeds_array[:,1] + yy1,
                                                          block_size, block_size, tc.batch_count)

            # glcm_features, cnn_features = tc.extract_cnn_feature(images)
            glcm_features, cnn_features = tc.extract_cnn_feature_with_iterator(img_itor)
            features = tc.comobine_features(glcm_features, cnn_features)

            predicted_tags = classifier.predict(features)
            mean_tags = np.mean(predicted_tags)
            print(index,'/',regions_count, "->", mean_tags, predicted_tags)
            result[self.ROI] = mean_tags

        return regions, result