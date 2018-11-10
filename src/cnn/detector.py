#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-26'

"""
from core import *
import numpy as np
from sklearn import metrics
from feature import FeatureExtractor
from skimage import io, util
from sklearn.svm import SVC
from sklearn.externals import joblib
from skimage import segmentation
from skimage.draw import rectangle # 需要skimage 0.14及以上版本
from core.util import get_seeds
import random
# from cnn import cnn_tensor
from cnn.cnn_simple_5x128 import cnn_simple_5x128

class Detector(object):

    def __init__(self, params, src_image):
        '''
        初始化
        :param params: 参数
        :param src_image: 切片图像
        '''
        self._params = params
        self._imgCone = src_image

        w, h = self._imgCone.get_image_width_height_byScale(self._params.GLOBAL_SCALE)
        self.ImageWidth = w
        self.ImageHeight = h
        self.valid_map = np.zeros((h, w), dtype=np.bool)

        return

    def setting_detected_area(self, x1, y1, x2, y2, scale):
        '''
        设置需要检测的区域
        :param x1: 左上角x坐标
        :param y1: 左上角y坐标
        :param x2: 右下角x坐标
        :param y2: 右下角y坐标
        :param scale: 以上坐标的倍镜数
        :return: 生成检测区的mask
        '''
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2 ])* GLOBAL_SCALE / scale).astype(np.int)
        rr, cc = rectangle((yy1, xx1), end=(yy2, xx2))
        self.valid_map[rr, cc] = 1
        self.valid_area_width = xx2 - xx1
        self.valid_area_height = yy2 - yy1
        return

    def reset_detected_area(self):
        '''
        清空检测区域的标记
        :return:
        '''
        self.valid_map =  np.zeros((self.ImageHeight, self.ImageWidth), dtype=np.bool)
        return

    def get_points_detected_area(self, extract_scale, patch_size_extract, spacing_extract):
        '''
        得到检测区域的图块中心点在高分辨率下的坐标
        :param extract_scale: 提取图时时，所使用的高分辨率对应的倍镜数
        :param patch_size_extract: 高分辨率下的图块大小
        :param spacing_extract: 高分辨率下的图块之间的距离
        :return: （x，y）种子点的集合
        '''
        return get_seeds(self.valid_map, self._params.GLOBAL_SCALE, extract_scale,patch_size_extract, spacing_extract, margin=8)

    def detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size):
        '''
        进行区域内的检测
        :param x1: 左上角x坐标
        :param y1: 左上角y坐标
        :param x2: 右下角x坐标
        :param y2: 右下角y坐标
        :param coordinate_scale:以上坐标的倍镜数
        :param extract_scale: 提取图块所用的倍镜
        :param patch_size: 图块大小
        :return: 图块中心点集，预测的结果
        '''
        self.setting_detected_area(x1, y1, x2, y2, coordinate_scale)
        seeds = self.get_points_detected_area(extract_scale, patch_size, patch_size >> 1)

        cnn = cnn_simple_5x128(self._params, "simplenet128")
        # predictions = cnn.predict(self._imgCone, extract_scale, patch_size, seeds)
        predictions = cnn.predict_on_batch(self._imgCone, extract_scale, patch_size, seeds, 20)

        return seeds, predictions

    def get_seed_deep_analysis(self, seeds, predictions, seeds_scale, original_patch_size, new_scale, new_patch_size):
        '''
        获取置信度不高的种子点在更高倍镜下的图块中心点坐标
        :param seeds: 低倍镜下的种子点集合
        :param predictions: 低倍镜下的种子点所对应的检测结果
        :param seeds_scale: 种子点的低倍镜数
        :param original_patch_size: 低倍镜下的图块大小
        :param new_scale: 高倍镜数
        :param new_patch_size: 在高倍镜下的图块大小
        :return: 高倍镜下，种子点集合
        '''
        amplify = new_scale / seeds_scale
        partitions = original_patch_size * amplify / new_patch_size
        bias = int(original_patch_size * amplify / (2 * partitions))
        result = []
        print(">>> Number of patches detected at low magnification: ", len(predictions))

        for (x, y), (class_id, probability) in zip(seeds, predictions):
            if probability < 0.95:
                xx = int(x * amplify)
                yy = int(y * amplify)
                result.append((xx, yy))
                result.append((xx - bias, yy - bias))
                result.append((xx - bias, yy + bias))
                result.append((xx + bias, yy - bias))
                result.append((xx + bias, yy + bias))

                # result.append((xx, yy - bias))
                # result.append((xx, yy + bias))
                # result.append((xx + bias, yy))
                # result.append((xx - bias, yy))
        print(">>> Number of patches to be detected at high magnification: ", len(result))
        return result

    def transform_coordinate(self, x1, y1, coordinate_scale, seeds_scale, target_scale, seeds):
        '''
        将图块中心坐标变换到新的坐标系中。 新坐标系的原点为检测区域的左上角，所处的倍镜为target_scale
        :param x1: 左上角x坐标
        :param y1: 左上角y坐标
        :param coordinate_scale: 以上坐标的倍镜数
        :param seeds_scale: 图块中心点（种子点）的倍镜
        :param target_scale: 目标坐标系所对应的倍镜
        :param seeds: 图块中心点集
        :return:新坐标系下的中心点
        '''
        xx1 = (x1 * target_scale / coordinate_scale)
        yy1 = (y1 * target_scale / coordinate_scale)

        results = []
        for x, y in seeds:
            xx = int(x * target_scale / seeds_scale - xx1)
            yy = int(y * target_scale / seeds_scale - yy1)
            # xx = max(0, xx)
            # yy = max(0, yy)
            results.append((xx, yy))
        # print(results)
        return results

    def create_cancer_map(self, x1, y1, coordinate_scale, seeds_scale, target_scale, seeds,
                          predictions, seeds_patch_size, pre_prob_map = None, pre_count_map = None):
        '''
        生成癌变可能性Map
        :param x1: 检测区域的左上角x坐标
        :param y1: 检测区域的左上角y坐标
        :param coordinate_scale: 以上坐标的倍镜数
        :param seeds_scale: 图块中心点（种子点）的倍镜
        :param target_scale: 目标坐标系所对应的倍镜
        :param seeds: 图块中心点集
        :param predictions: 每个图块的预测结果
        :param seeds_patch_size: 图块的大小
        :param pre_prob_map: 上次处理生成癌变概率图
        :param pre_count_map; 上次处理生成癌变的检测计数图
        :return: 癌变可能性Map
        '''
        new_seeds = self.transform_coordinate(x1, y1, coordinate_scale, seeds_scale, target_scale, seeds)
        target_patch_size = int(seeds_patch_size * target_scale / seeds_scale)
        half = int(target_patch_size>>1)

        cancer_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)
        prob_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)
        count_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)

        update_mode = not (pre_prob_map is None or pre_count_map is None)
        if update_mode:
            vaild_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.bool)

        for (x, y), (class_id, probability)  in zip(new_seeds, predictions):
            xx = x - half
            yy = y - half
            rr, cc = rectangle((yy, xx), extent=(target_patch_size, target_patch_size))

            select_y = (rr >= 0) & (rr < self.valid_area_height)
            select_x = (cc >= 0) & (cc < self.valid_area_width)
            select = select_x & select_y

            if class_id == 1 :
                prob_map[rr[select], cc[select]] = prob_map[rr[select], cc[select]] + probability
                count_map[rr[select], cc[select]] = count_map[rr[select], cc[select]] + 1

            if update_mode:
                vaild_map[rr[select], cc[select]] = True

        tag = count_map > 0
        cancer_map[tag] = prob_map[tag] / count_map[tag]

        if not update_mode:
            #计算平均概率
            return cancer_map, prob_map, count_map
        else:
            # 更新平均概率
            pre_prob_map[vaild_map] = prob_map[vaild_map]
            pre_count_map[vaild_map] = count_map[vaild_map]

            tag = pre_count_map > 0
            cancer_map[tag] = pre_prob_map[tag] / pre_count_map[tag]
            return cancer_map, pre_prob_map, pre_count_map

    def get_detect_area_img(self, x1, y1, x2, y2, coordinate_scale, img_scale):
        '''
        得到指定的检测区域对应的图像
        :param x1: 左上角x坐标
        :param y1: 左上角y坐标
        :param x2: 右下角x坐标
        :param y2: 右下角y坐标
        :param coordinate_scale:以上坐标的倍镜数
        :param img_scale: 提取图像所对应的倍镜
        :return:指定的检测区域对应的图像
        '''
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * img_scale / coordinate_scale).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1
        block = self._imgCone.get_image_block(img_scale, int(xx1 + (w >> 1)), int(yy1 + (h >> 1)), w, h)
        return block.get_img()