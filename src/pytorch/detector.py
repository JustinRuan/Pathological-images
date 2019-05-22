#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-26'

"""
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from scipy.interpolate import griddata
from skimage import morphology
from skimage.draw import rectangle  # 需要skimage 0.14及以上版本
from skimage.measure import find_contours
from skimage.morphology import square, dilation, erosion
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from visdom import Visdom

from core import Random_Gen
from core.util import get_seeds, transform_coordinate
from pytorch.cnn_classifier import Simple_Classifier, MultiTask_Classifier, MSC_Classifier
from pytorch.segmentation import Segmentation
from pytorch.transfer_cnn import Transfer
from preparation.normalization import HistNormalization


class BaseDetector(object, metaclass=ABCMeta):
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
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)
        rr, cc = rectangle((yy1, xx1), end=(yy2, xx2))
        self.valid_map[rr, cc] = 1
        self.valid_area_width = xx2 - xx1
        self.valid_area_height = yy2 - yy1
        self.valid_rect = (xx1, yy1, xx2, yy2)
        return

    def reset_detected_area(self):
        '''
        清空检测区域的标记
        :return:
        '''
        self.valid_map = np.zeros((self.ImageHeight, self.ImageWidth), dtype=np.bool)
        return

    def evaluate(self, threshold, cancer_map, true_mask):
        '''
        癌变概率矩阵进行阈值分割后，与人工标记真值进行 评估
        :param threshold: 分割的阈值
        :param cancer_map: 癌变概率矩阵
        :param true_mask: 人工标记真值
        :return: ROC曲线
        '''
        cancer_tag = np.array(cancer_map).ravel()
        mask_tag = np.array(true_mask).ravel()
        predicted_tags = cancer_tag >= threshold

        print("Classification report for classifier:\n%s\n"
              % (metrics.classification_report(mask_tag, predicted_tags, digits=4)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(mask_tag, predicted_tags))

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(mask_tag, cancer_tag)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        print("\n ROC auc: %s" % roc_auc)

        dice = self.calculate_dice_coef(mask_tag, cancer_tag)
        print("dice coef = {}".format(dice))
        print("############################################################")
        return false_positive_rate, true_positive_rate, roc_auc, dice

    def calculate_dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)
        return dice

    def get_img_in_detect_area(self, x1, y1, x2, y2, coordinate_scale, img_scale):
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

    def get_true_mask_in_detect_area(self, x1, y1, x2, y2, coordinate_scale, img_scale):
        '''
        生成选定区域内的人工标记的Mask
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :param coordinate_scale 以上坐标的倍镜数:
        :param img_scale: 生成Mask图像的倍镜数
        :return: mask图像
        '''
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * img_scale / coordinate_scale).astype(np.int)
        w = xx2 - xx1
        h = yy2 - yy1

        all_mask = self._imgCone.create_mask_image(img_scale, 0)
        cancer_mask = all_mask['C']
        return cancer_mask[yy1:yy2, xx1:xx2]

    @abstractmethod
    def process(self, x1, y1, x2, y2, coordinate_scale, **kwargs):
        pass

class Detector(BaseDetector):
    def __init__(self, params, src_image):
        super(Detector, self).__init__(params, src_image)
        self.enable_transfer = True

    def get_points_detected_area(self, extract_scale, patch_size_extract, interval):
        '''
        得到检测区域的图块中心点在高分辨率下的坐标
        :param extract_scale: 提取图时时，所使用的高分辨率对应的倍镜数
        :param patch_size_extract: 高分辨率下的图块大小
        :param interval: 高分辨率下的图块之间的距离
        :return: （x，y）种子点的集合
        '''
        return get_seeds(self.valid_map, self._params.GLOBAL_SCALE, extract_scale, patch_size_extract, interval,
                         margin=8)

    def detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size, interval):
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
        seeds = self.get_points_detected_area(extract_scale, patch_size, interval)

        if self.enable_transfer:
            #####################################################################################################
            #    Transfer Learning
            #####################################################################################################
            cnn = Transfer(self._params, "densenet121", "500_128")
        #########################################################################################################
        else:
            ########################################################################################################\
            #    DenseNet 22
            #########################################################################################################
            cnn = Simple_Classifier(self._params, "densenet_22", "500_128")
        #########################################################################################################
        predictions = cnn.predict_on_batch(self._imgCone, extract_scale, patch_size, seeds, 32)

        return seeds, predictions

    def detect_region_detailed(self, seeds, predictions, seeds_scale, original_patch_size, new_scale, new_patch_size):
        new_seeds = self.get_seeds_under_high_magnification(seeds, predictions, seeds_scale, original_patch_size,
                                                            new_scale, new_patch_size)

        if self.enable_transfer:
            #####################################################################################################
            #    Transfer Learning
            #####################################################################################################
            if (new_scale == 20):
                cnn = Transfer(self._params, "densenet121", "2000_256")
            else:  # (new_scale == 40):
                cnn = Transfer(self._params, "densenet121", "4000_256")
        #########################################################################################################
        else:
            ########################################################################################################\
            #    DenseNet 22
            #########################################################################################################
            if (new_scale == 20):
                cnn = Simple_Classifier(self._params, "densenet_22", "2000_256")
            else:  # (new_scale == 40):
                cnn = Simple_Classifier(self._params, "densenet_22", "4000_256")
        #########################################################################################################
        predictions = cnn.predict_on_batch(self._imgCone, new_scale, new_patch_size, new_seeds, 32)
        return new_seeds, predictions

    def get_seeds_under_high_magnification(self, seeds, predictions, seeds_scale, original_patch_size, new_scale,
                                           new_patch_size):
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

    def create_cancer_map(self, x1, y1, coordinate_scale, seeds_scale, target_scale, seeds,
                          predictions, seeds_patch_size, pre_prob_map=None, pre_count_map=None):
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
        :param seed_interval:
        :param pre_prob_map: 上次处理生成癌变概率图
        :param pre_count_map; 上次处理生成癌变的检测计数图
        :return: 癌变可能性Map
        '''
        new_seeds = transform_coordinate(x1, y1, coordinate_scale, seeds_scale, target_scale, seeds)
        target_patch_size = int(seeds_patch_size * target_scale / seeds_scale)
        half = int(target_patch_size >> 1)

        cancer_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)
        prob_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)
        count_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)

        update_mode = not (pre_prob_map is None or pre_count_map is None)
        if update_mode:
            vaild_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.bool)

        for (x, y), (class_id, probability) in zip(new_seeds, predictions):
            xx = x - half
            yy = y - half

            for K in np.logspace(0, 2, 3, base=2):  # array([1., 2., 4.])
                w = int(target_patch_size / K)
                if w == 0:
                    continue

                rr, cc = rectangle((yy, xx), extent=(w, w))

                select_y = (rr >= 0) & (rr < self.valid_area_height)
                select_x = (cc >= 0) & (cc < self.valid_area_width)
                select = select_x & select_y

                if class_id == 1:
                    prob_map[rr[select], cc[select]] = prob_map[rr[select], cc[select]] + probability
                else:
                    prob_map[rr[select], cc[select]] = prob_map[rr[select], cc[select]] + (1 - probability)

                count_map[rr[select], cc[select]] = count_map[rr[select], cc[select]] + 1

                if update_mode:
                    vaild_map[rr[select], cc[select]] = True

        if update_mode:
            pre_cancer_map = np.zeros((self.valid_area_height, self.valid_area_width), dtype=np.float)
            tag = pre_count_map > 0
            pre_cancer_map[tag] = pre_prob_map[tag] / pre_count_map[tag]

            # 更新平均概率
            keep_tag = (~vaild_map) | (pre_cancer_map >= 0.8)  # 提高低倍镜分类器性能，将可以提高这个阈值
            prob_map[keep_tag] = pre_prob_map[keep_tag]
            count_map[keep_tag] = pre_count_map[keep_tag]

        tag = count_map > 0
        cancer_map[tag] = prob_map[tag] / count_map[tag]
        return cancer_map, prob_map, count_map

    def create_superpixels(self, x1, y1, x2, y2, coordinate_scale, feature_extract_scale):
        '''
        在指定区域内，提取特征，并进行超像素分割
        :param x1: 左上角x
        :param y1: 左上角y
        :param x2: 右下角x
        :param y2: 右下角y
        :param coordinate_scale: 以上坐标的倍镜数
        :param feature_extract_scale: 提取特征所用的倍镜数
        :return: 超像素分割后的标记矩阵
        '''
        seg = Segmentation(self._params, self._imgCone)
        f_map = seg.create_feature_map(x1, y1, x2, y2, coordinate_scale, feature_extract_scale)
        label_map = seg.create_superpixels(f_map, 0.4, iter_num=3)

        return label_map

    def create_cancer_map_superpixels(self, cancer_map, label_map):
        '''
        根据超像素分割，生成癌变概率图
        :param cancer_map: 输入的癌变概率图
        :param label_map:超像素分割的结果
        :return: 融合后生成的癌变概率图
        '''
        label_set = set(label_map.flatten())

        result_cancer = np.zeros(cancer_map.shape, dtype=np.float)
        for label_id in label_set:
            roi = (label_map == label_id)
            cancer_roi = cancer_map[roi]
            mean = np.mean(cancer_roi, axis=None)
            std = np.std(cancer_roi, axis=None)

            result_cancer[roi] = mean

        return result_cancer

    ##################################################################################################################
    ##########   自适应采样，全切片扫描    #################
    ##################################################################################################################

    # 第一版本
    # def adaptive_detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size,
    #                            max_iter_nums, batch_size, use_post=True):
    #     self.setting_detected_area(x1, y1, x2, y2, coordinate_scale)
    #     print("h = ", self.valid_area_height, ", w = ", self.valid_area_width)
    #
    #     # cnn = CNN_Classifier(self._params, "densenet_22", "2000_256")
    #     cnn = CNN_Classifier(self._params, "se_densenet_22", "x_256")
    #
    #     # 生成坐标网格
    #     grid_y, grid_x = np.mgrid[0: self.valid_area_height: 1, 0: self.valid_area_width: 1]
    #
    #     sobel_img = None
    #     interpolate_img = None
    #     threshold = 0.05
    #     history = {}
    #     N = 400
    #
    #     seeds_scale = self._params.GLOBAL_SCALE
    #
    #     #########################################################################################################
    #     viz = Visdom(env="main")
    #     pic_thresh = None
    #     pic_points = None
    #     mask_img = self.get_true_mask_in_detect_area(x1, y1, x2, y2, coordinate_scale, seeds_scale)
    #     c_mask = find_contours(np.array(mask_img).astype(int), level=0.5)
    #     for i, contour in enumerate(c_mask):
    #         contour = np.abs(np.array(contour - [y2 - y1, 0]))
    #         c_name = "GT {}".format(i)
    #         if pic_points is None:
    #             pic_points = viz.line(Y=contour[:, 0], X=contour[:, 1], name=c_name,
    #                                   opts={'linecolor': np.array([[0, 0, 0], ]), 'showlegend': True, })
    #         else:
    #             viz.line(Y=contour[:, 0], X=contour[:, 1], name=c_name, win=pic_points, update='append',
    #                      opts={'linecolor': np.array([[0, 0, 0], ])})
    #     #########################################################################################################
    #
    #     for i in range(max_iter_nums):
    #         print("iter %d" % (i + 1))
    #         seeds = self.get_random_seeds(N, x1, x2, y1, y2, sobel_img, threshold)
    #
    #         new_seeds = self.remove_duplicates(x1, y1, seeds, set(history.keys()))
    #         print("the number of new seeds: ", len(new_seeds))
    #         #######################################################################################
    #         t_seeds = np.abs(np.array(list(new_seeds)) - [x1, y2])  # 坐标原点移动，并翻转
    #         len_seed = len(new_seeds)
    #         t_y = np.full((len_seed, 1), i + 1)
    #         random_color = np.tile(np.random.randint(0, 255, (1, 3,)), (len_seed, 1))
    #         step_name = "Round {}".format(i + 1)
    #         viz.scatter(X=t_seeds, Y=t_y, name=step_name, win=pic_points, update="append",
    #                     opts=dict(title='seeds', caption='seeds', showlegend=True,
    #                               markercolor=random_color, markersize=8))
    #         ########################################################################################
    #         if len(new_seeds) / N < 0.9:
    #             break
    #
    #         high_seeds = transform_coordinate(0, 0, coordinate_scale, seeds_scale, extract_scale, new_seeds)
    #         predictions = cnn.predict_on_batch(self._imgCone, extract_scale, patch_size, high_seeds, batch_size)
    #         probs = self.get_cancer_probability(predictions)
    #
    #         for (x, y), pred in zip(new_seeds, probs):
    #             xx = x - x1
    #             yy = y - y1
    #
    #             if not history.__contains__((xx, yy)):
    #                 history[(xx, yy)] = pred
    #
    #         value = list(history.values())
    #         point = list(history.keys())
    #         # interpolate_img, sobel_img = self.inter_sobel(point, value,
    #         #                                               (grid_x, grid_y), method='linear')
    #         # 使用cubic，会出现负值，而选用linear不会这样
    #         interpolate_img = griddata(point, value, (grid_x, grid_y), method='linear', fill_value=0.0)
    #         sobel_img, threshold = self.calc_sobel(interpolate_img)
    #
    #         ########################################################################################################
    #
    #         if pic_thresh is None:
    #             pic_thresh = viz.line(Y=[threshold], X=[i], opts=dict(title='treshold', caption='treshold'))
    #         else:
    #             viz.line(Y=[threshold], X=[i], win=pic_thresh, update="append")
    #
    #         #########################################################################################################
    #
    #     if use_post:
    #         amplify = extract_scale / seeds_scale
    #         bias = int(0.25 * patch_size / amplify)
    #         interpolate_img = self.post_process(interpolate_img, bias)
    #
    #     np.savez("detect.npz", interpolate_img, history)
    #     return interpolate_img, history

    # def get_random_seeds(self, N, x1, x2, y1, y2, sobel_img, threshold):
    #     if sobel_img is not None and threshold > 0.015:
    #         x = []
    #         y = []
    #         while len(x) < N:
    #             n = 2 * N
    #             sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)
    #
    #             prob = sobel_img[sy - y1, sx - x1]
    #             index = prob >= threshold
    #             sx = sx[index]
    #             sy = sy[index]
    #
    #             x.extend(sx)
    #             y.extend(sy)
    #     else:
    #         n = N
    #         x, y = self.random_gen.generate_random(n, x1, x2, y1, y2)
    #     return tuple(zip(x, y))

    # def calc_sobel(self, interpolate):
    #     sobel_img = np.abs(cv2.Sobel(interpolate, -1, 1, 1))
    #     sobel_value = sobel_img.reshape(-1, 1)
    #
    #     clustering = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100,
    #                                  batch_size=1000, tol=1e-3).fit(sobel_value)
    #
    #     self.cluster_centers = clustering.cluster_centers_.ravel()
    #     threshold = np.mean(self.cluster_centers)
    #     print("threshold = {:.6f}, clustering = {}".format(threshold, self.cluster_centers))
    #
    #     # threshold小于设定值时，膨胀的作用在于，增加大梯点的数量，以便更容易找到这些区域。
    #     #                        同时这样也能避免算法过早收敛。（超过阈值的点太少，找不到新的就收敛了）
    #     # threshold大于设定值时，说明已经发现了高梯度区域，不进行膨胀操作，可以加速算法收敛。
    #     #                        但Dice会有所降低。这里只能折中考虑了。
    #     # if threshold < self.search_therhold:
    #     #     sobel_img = dilation(sobel_img, square(8))
    #
    #     return sobel_img, threshold
    def process(self, x1, y1, x2, y2, coordinate_scale, **kwargs):
        interval = kwargs["interval"]

        seeds, predictions = self.detect_region(x1, y1, x2, y2, 1.25, 5, 128, interval=interval)
        new20_seeds, new20_predictions = self.detect_region_detailed(seeds, predictions, 5, 128, 20, 256)
        new40_seeds, new40_predictions = self.detect_region_detailed(new20_seeds, new20_predictions, 20, 256, 40,
                                                                         256)

        cancer_map, prob_map, count_map = self.create_cancer_map(x1, y1, 1.25, 5, 1.25, seeds, predictions, 128,
                                                                     None, None)
        cancer_map2, prob_map, count_map = self.create_cancer_map(x1, y1, 1.25, 20, 1.25, new20_seeds,
                                                                      new20_predictions,
                                                                      256, prob_map, count_map)
        cancer_map3, prob_map, count_map = self.create_cancer_map(x1, y1, 1.25, 40, 1.25, new40_seeds,
                                                                      new40_predictions,
                                                                      256, prob_map, count_map)

        seg = Segmentation(self._params, self._imgCone)
        label_map = seg.create_superpixels_slic(x1, y1, x2, y2, 1.25, 1.25, 30, 20)

        cancer_map4 = self.create_cancer_map_superpixels(cancer_map3, label_map)

        return cancer_map, cancer_map2, cancer_map3, cancer_map4


class AdaptiveDetector(BaseDetector):
    def __init__(self, params, src_image):
        super(AdaptiveDetector, self).__init__(params, src_image)

        self.random_gen = Random_Gen("halton")  # random, sobol, halton
        self.cluster_centers = None

        self.search_therhold = 0.015

    def get_cancer_probability(self, predictions):
        '''
        计算出每个检测点所得到的癌变概率
        :param predictions: 预测的结果列表，（pred 预测的类型, prob可能性）
        :return:概率列表
        '''
        probs = []
        for pred, prob in predictions:
            if pred == 0:
                probs.append(1 - prob)
            else:
                probs.append(prob)

        return probs

    def remove_duplicates(self, x1, y1, new_seeds, old_seeds):
        '''
        从新生成的种子坐标中，排除已经检测过的坐标位置
        :param x1: 当前种子点new_seeds和old_seeds所用坐标系的原点，在全切片中的绝对x坐标
        :param y1: 当前种子点new_seeds和old_seeds所用坐标系的原点，在全切片中的绝对y坐标
        :param new_seeds: 新种子坐标（全切片中的绝对坐标，原点为切片的左上角）
        :param old_seeds: 以前的检测过的坐标（在当前检测区域中的坐标，原点为检测区域的左上角）
        :return: 没有检测过的种子点坐标，（原点为切片的左上角）
        '''
        shift_seeds = set((xx - x1, yy - y1) for xx, yy in new_seeds)
        result = shift_seeds - old_seeds
        revert_seeds = set((xx + x1, yy + y1) for xx, yy in result)
        return revert_seeds

    def get_random_seeds(self, N, x0, y0, x1, x2, y1, y2, sobel_img, threshold):
        '''
        获取随机的种子点坐标（基本算法）
        :param N: 获取种子点的数量
        :param x0: Sobel_image的左上角在全切片的绝对位置坐标x
        :param y0: Sobel_image的左上角在全切片的绝对位置坐标y
        :param x1: 在全切片中当前检测区域的左上角x
        :param x2: 在全切片中当前检测区域的右下角x
        :param y1: 在全切片中当前检测区域的左上角y
        :param y2: 在全切片中当前检测区域的右下角y
        :param sobel_img: 梯度图
        :param threshold: 边界阈值
        :return: 种子点
        '''
        if sobel_img is not None and threshold > self.search_therhold:
            x = []
            y = []
            count = 0
            while len(x) < N:
                n = 2 * N
                sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)

                prob = sobel_img[sy - y0, sx - x0]
                index = prob >= threshold
                sx = sx[index]
                sy = sy[index]

                x.extend(sx)
                y.extend(sy)
                count += 1
                # 这里设定了尝试的次数上限
                # 设定越大，算法收敛速度越慢，但效果会好一点，。
                if count > 50:
                    break
        else:
            n = N
            x, y = self.random_gen.generate_random(n, x1, x2, y1, y2)
        return tuple(zip(x, y))

    def post_process(self, cancer_map, bias, thresh=(0.7, 0.5)):
        '''
        后处理，对低概率区域进行抑制，对高概率区域进行膨胀
        :param cancer_map: 癌症概率 图
        :param bias: 形态学运算的模板大小
        :param thresh: 高低概率的阈值
        :return: 概率图
        '''
        high_region = cancer_map > thresh[0] # 高概率
        low_region = cancer_map > thresh[1] # 低概率

        candidated_tag, num_tag = morphology.label(low_region, neighbors=8, return_num=True)

        for index in range(1, num_tag + 1):
            selected_region = candidated_tag == index
            total = np.sum(high_region[selected_region] == True)

            # 高概率区域面积过小，则抑制
            if total < 2 * 64:  # 256 / (40 /1.25) = 8
                selected_cancer_map = cancer_map.copy()
                temp = erosion(selected_cancer_map, square(2 * bias))
                temp = erosion(temp, square(2 * bias))
                cancer_map[selected_region] = temp[selected_region]

        temp = erosion(cancer_map, square(bias))
        result = dilation(temp, square(bias))
        result = dilation(result, square(bias))

        return result

    def process(self, x1, y1, x2, y2, coordinate_scale, **kwargs):
        extract_scale = kwargs["extract_scale"]
        patch_size = kwargs["patch_size"]
        max_iter_nums = kwargs["max_iter_nums"]
        batch_size = kwargs["batch_size"]
        limit_sampling_density = kwargs["limit_sampling_density"]

        return self.adaptive_detect_region(x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size,
                               max_iter_nums, batch_size, limit_sampling_density, use_post=True)

    # 第三版本: 增强自适应采样
    def adaptive_detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size,
                               max_iter_nums, batch_size, limit_sampling_density = 8.0, use_post=True):
        '''

        :param x1: 检测区域的左上角x
        :param y1: 检测区域的左上角y
        :param x2: 检测区域的右下角x
        :param y2: 检测区域的右下角y
        :param coordinate_scale: （x1, y1, x2, y2）所对应的倍镜数
        :param extract_scale: 提到图块所用的倍镜数
        :param patch_size: 图块边长
        :param max_iter_nums: 最大的迭代轮数
        :param batch_size: 每批次的图片数量
        :param use_post: 是否使用后处理操作
        :return:
        '''
        self.setting_detected_area(x1, y1, x2, y2, coordinate_scale)
        print("h = ", self.valid_area_height, ", w = ", self.valid_area_width)
        # self.effi_seeds = self.get_effective_seeds(x1, y1, x2, y2, coordinate_scale, self._params.GLOBAL_SCALE)
        seeds_scale = self._params.GLOBAL_SCALE

        seg = Segmentation(self._params, self._imgCone)
        # region_count = self.valid_area_height * self.valid_area_width // 400
        region_count = 30
        print("the number of superpixel regions ", region_count)

        label_map = seg.create_superpixels_slic(x1, y1, x2, y2, coordinate_scale, seeds_scale, region_count, 20)
        boundary_seeds = seg.get_seeds_at_boundaries(label_map, x1, y1, coordinate_scale, 64)
        print("the number of seeds at boundaries is ", len(boundary_seeds))

        normal_func = None
        # normal_func = HistNormalization("match_hist", hist_target ="hist_templates.npy", hist_source = None)

        # model_name = "se_densenet_22"
        # sample_name = "x_256"
        # cnn = MultiTask_Classifier(self._params, model_name, sample_name, normalization=normal_func)

        #         # sample_name = "msc_256"
        # model_name = "se_densenet_22"
        # model_name = "se_densenet_c9_22"

        # model_name = "simple_cnn"
        # model_name = "se_densenet_40"
        model_name = "densenet_22"
        sample_name = "4000_256"
        cnn = Simple_Classifier(self._params, model_name, sample_name, normalization=normal_func)

        # 生成坐标网格
        grid_y, grid_x = np.mgrid[0: self.valid_area_height: 1, 0: self.valid_area_width: 1]

        sobel_img = None
        interpolate_img = None
        history = {}
        N = 400

        #########################################################################################################
        viz = Visdom(env="main")
        pic_thresh = None
        pic_points = None
        pic_density = None
        mask_img = self.get_true_mask_in_detect_area(x1, y1, x2, y2, coordinate_scale, seeds_scale)
        c_mask = find_contours(np.array(mask_img).astype(int), level=0.5)
        for i, contour in enumerate(c_mask):
            contour = np.abs(np.array(contour - [y2 - y1, 0]))
            c_name = "GT {}".format(i)
            if pic_points is None:
                pic_points = viz.line(Y=contour[:, 0], X=contour[:, 1], name=c_name,
                                      opts={'linecolor': np.array([[0, 0, 0], ]), 'showlegend': True, })
            else:
                viz.line(Y=contour[:, 0], X=contour[:, 1], name=c_name, win=pic_points, update='append',
                         opts={'linecolor': np.array([[0, 0, 0], ])})
        #########################################################################################################

        threshold = 0.0 # 边缘区域与平坦区域的分割阈值
        total_step = 1

        rx1, ry1, rx2, ry2 = self.valid_rect
        for i in range(max_iter_nums):

            print("iter {}, {}, {}".format(i + 1, (rx1, ry1), (rx2, ry2)))
            # seeds = self.get_random_seeds(N, x1, y1, rx1, rx2, ry1, ry2, sobel_img, threshold)
            seeds = self.get_random_seeds_ex2(N, x1, y1, rx1, rx2, ry1, ry2, sobel_img, threshold)
            if i == 0:
                seeds.extend(boundary_seeds)

            new_seeds = self.remove_duplicates(x1, y1, seeds, set(history.keys()))
            print("the number of new seeds: ", len(new_seeds), ', the number of seeds in history:', len(history))

            sampling_density = self.cacl_sampling_density(x1, y1, new_seeds, list(history.keys()), r=8)
            print("Current sampling density = ", sampling_density)
            ########################################################################################
            if pic_density is None:
                pic_density = viz.line(Y=[sampling_density], X=[total_step], opts=dict(title='sampling density', caption='sampling density'))
            else:
                pic_density = viz.line(Y=[sampling_density], X=[total_step], win=pic_density, update="append")
            ########################################################################################

            # 单倍镜下进行检测
            if model_name in ["se_densenet_40", "se_densenet_22", "densenet_22", "simple_cnn"]:
                high_seeds = transform_coordinate(0, 0, coordinate_scale, seeds_scale, extract_scale, new_seeds)
                predictions = cnn.predict_on_batch(self._imgCone, extract_scale, patch_size, high_seeds, batch_size)
                probs = self.get_cancer_probability(predictions)
            # elif model_name == "simple_cnn":
            #     high_seeds = transform_coordinate(0, 0, coordinate_scale, seeds_scale, extract_scale, new_seeds)
            #     predictions = cnn.predict_on_batch(self._imgCone, extract_scale, patch_size, high_seeds, batch_size)
            #     probs = self.get_cancer_probability(predictions)
            # # 多倍镜下进行检测, 效果不好啊！
            # elif mode == 2:
            #     # def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
            #     probs = cnn.predict_multi_scale(self._imgCone, (10, 20, 40), 256, seeds_scale, new_seeds, batch_size)
            # 多倍镜联合
            elif model_name == "se_densenet_c9_22":
                probs = cnn.predict_msc(self._imgCone, 256, seeds_scale, new_seeds, batch_size)
                # probs = self.get_cancer_probability(predictions)

            #######################################################################################
            t_seeds = np.abs(np.array(list(new_seeds)) - [x1, y2])  # 坐标原点移动，并翻转
            len_seed = len(new_seeds)
            random_color = np.tile(np.random.randint(0, 255, (1, 3,)), (len_seed, 1))
            step_name = "Round {}".format(total_step)
            text_labels = []
            for item in probs:
                text_labels.append("{:.2f}".format(item))

            viz.scatter(X=t_seeds, name=step_name, win=pic_points, update="append",
                        opts=dict(title='seeds', caption='seeds', showlegend=True,  #textlabels=text_labels,
                                  markercolor=random_color, markersize=8))
            ########################################################################################

            for (x, y), pred in zip(new_seeds, probs):
                xx = x - x1
                yy = y - y1

                if not history.__contains__((xx, yy)):
                    history[(xx, yy)] = pred

            value = list(history.values())
            point = list(history.keys())

            # 使用cubic，会出现负值，而选用linear不会这样
            interpolate_img = griddata(point, value, (grid_x, grid_y), method='linear', fill_value=0.0)
            sobel_img, threshold = self.calc_sobel(interpolate_img)

            ########################################################################################################

            if pic_thresh is None:
                pic_thresh = viz.line(Y=[threshold], X=[total_step], opts=dict(title='treshold', caption='treshold'))
            else:
                pic_thresh = viz.line(Y=[threshold], X=[total_step], win=pic_thresh, update="append")

            # if pic_thresh is None:
            #     pic_thresh = viz.line(Y=np.column_stack((threshold, sampling_density)), X=np.column_stack((total_step, total_step)), opts=dict(title='treshold', caption='treshold'))
            # else:
            #     viz.line(Y=np.column_stack((threshold, sampling_density)), X=np.column_stack((total_step, total_step)), win=pic_thresh, update="append")
            #########################################################################################################
            total_step += 1

            # if (len(new_seeds) / N < 0.9):
            #     # 避免过早收敛
            #     count_tresh += 1
            #     if count_tresh >= 2:
            #         break
            # else:
            #     count_tresh = 0
            if sampling_density > limit_sampling_density:
                break

        if use_post:
            amplify = extract_scale / seeds_scale
            bias = int(0.25 * patch_size / amplify)
            interpolate_img = self.post_process(interpolate_img, bias)

        # np.savez("detect.npz", interpolate_img, history)
        return interpolate_img, history

    def cacl_sampling_density(self, x1, y1, new_seeds, old_seeds, r = 8):

        if not old_seeds:
            return 0.0

        seeds = np.array(old_seeds)
        shift_seeds = set((xx - x1, yy - y1) for xx, yy in new_seeds)
        result = []

        for nx, ny in shift_seeds:
            dxy = seeds - np.array([nx, ny])
            max_dxy = np.max(np.abs(dxy), axis=1)
            count = np.sum(max_dxy < r)
            result.append(count)

        return np.mean(result)

    def calc_sobel(self, interpolate):
        '''
        计算梯度图，和边缘区域的阈值
        :param interpolate 概率图:
        :return: 梯度图，和边缘区域的阈值
        '''
        sobel_img = np.abs(cv2.Sobel(interpolate, -1, 1, 1))
        sobel_value = sobel_img.reshape(-1, 1)

        clustering = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100, compute_labels=False,
                                     batch_size=1000, tol=1e-3).fit(sobel_value)

        self.cluster_centers = clustering.cluster_centers_.ravel()
        threshold = np.mean(self.cluster_centers)
        print("threshold = {:.6f}, clustering = {}".format(threshold, self.cluster_centers))
        return sobel_img, threshold

    def get_random_seeds_ex(self, N, x0, y0, x1, x2, y1, y2, sobel_img, threshold):
        '''
        获取随机的种子点坐标（扩展算法）
        :param N: 获取种子点的数量
        :param x0: Sobel_image的左上角在全切片的绝对位置坐标x
        :param y0: Sobel_image的左上角在全切片的绝对位置坐标y
        :param x1: 在全切片中当前检测区域的左上角x
        :param x2: 在全切片中当前检测区域的右下角x
        :param y1: 在全切片中当前检测区域的左上角y
        :param y2: 在全切片中当前检测区域的右下角y
        :param sobel_img: 梯度图
        :param threshold: 边界阈值
        :return: 种子点
        '''
        # 目前，自适应采样分为三个阶段进行：
        # 1）第一轮，全局的Haltan随机采样，以此生成Sobel图
        # 2）第二轮开始，直到检测阈值Threshold达到设定范围之前的阶段：根据WxW区域内梯度局部极大值进行种子点的选择。
        #       相当于图像分辨率下降后的随机搜索过程，加速寻找可能性的区域。
        # 3）检测阈值达到设定值Threshold后， 算法进入精确的搜索阶段：只选择梯度值越过设定Threshold的点进行采样，
        #       直到采样过密，采样点重合条件达到，退出算法。
        if sobel_img is None:  # 第一轮
            n = N
            x, y = self.random_gen.generate_random(n, x1, x2, y1, y2)
        else:
            x = []
            y = []
            if threshold > self.search_therhold:  # 精确搜索开始
                count = 0
                while len(x) < N:
                    n = 2 * N + count * N
                    sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)

                    prob = sobel_img[sy - y0, sx - x0]
                    index = prob >= threshold
                    sx = sx[index]
                    sy = sy[index]

                    x.extend(sx)
                    y.extend(sy)
                    count += 1
                    # 这里设定了尝试的次数上限
                    # 设定越大，算法收敛速度越慢，但效果会好一点，。
                    if count > 50:
                        break
            else:  # 大范围地随机搜索
                w = 16
                half_w = w >> 1
                n = 2 * N
                sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)
                grad_list = []
                for xx, yy in zip(sx, sy):
                    rr, cc = rectangle((yy - half_w - y0, xx - half_w - x0), extent=(w, w))

                    select_y = (rr >= 0) & (rr < self.valid_area_height)
                    select_x = (cc >= 0) & (cc < self.valid_area_width)
                    select = select_x & select_y
                    max_grad = np.max(sobel_img[rr[select], cc[select]])
                    grad_list.append(max_grad)

                index = np.array(grad_list).argsort()
                sx = sx[index[-N:]]
                sy = sy[index[-N:]]

                x.extend(sx)
                y.extend(sy)

        return tuple(zip(x, y))

    # 贪婪策略
    def get_random_seeds_ex2(self, N, x0, y0, x1, x2, y1, y2, sobel_img, threshold):
        '''
        获取随机的种子点坐标（扩展算法）
        :param N: 获取种子点的数量
        :param x0: Sobel_image的左上角在全切片的绝对位置坐标x
        :param y0: Sobel_image的左上角在全切片的绝对位置坐标y
        :param x1: 在全切片中当前检测区域的左上角x
        :param x2: 在全切片中当前检测区域的右下角x
        :param y1: 在全切片中当前检测区域的左上角y
        :param y2: 在全切片中当前检测区域的右下角y
        :param sobel_img: 梯度图
        :param threshold: 边界阈值
        :return: 种子点
        '''
        # 目前，自适应采样分为三个阶段进行：
        # 1）第一轮，全局的Haltan随机采样，以此生成Sobel图
        # 2）第二轮开始，直到检测阈值Threshold达到设定范围之前的阶段：根据WxW区域内梯度局部极大值进行种子点的选择。
        #       相当于图像分辨率下降后的随机搜索过程，加速寻找可能性的区域。
        # 3）检测阈值达到设定值Threshold后， 算法进入精确的搜索阶段：只选择梯度值越过设定Threshold的点进行采样，
        #       直到采样过密，采样点重合条件达到，退出算法。
        if sobel_img is None:  # 第一轮
            n = N
            x, y = self.random_gen.generate_random(n, x1, x2, y1, y2)
        else:
            x = []
            y = []
            if threshold > self.search_therhold:  # 精确搜索开始
                n = 4 * N
                while len(x) <= N/10:
                    sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)

                    prob = sobel_img[sy - y0, sx - x0]
                    index = prob >= threshold
                    sx = sx[index]
                    sy = sy[index]

                    x.extend(sx)
                    y.extend(sy)

                half_w = 128

                if len(x) < N:
                    for xx, yy in zip(sx, sy):
                        nx1 = max(xx - half_w, x1)
                        nx2 = min(xx + half_w, x2)
                        ny1 = max(yy - half_w, y1)
                        ny2 = min(yy + half_w, y2)
                        sx2, sy2 = self.random_gen.generate_random(n, nx1, nx2, ny1, ny2)

                        prob = sobel_img[sy2 - y0, sx2 - x0]
                        index = prob >= threshold
                        sx = sx2[index]
                        sy = sy2[index]

                        x.extend(sx)
                        y.extend(sy)

                # 从当前的x和y的结果中只选择N个
                m = len(x)
                if m > N:
                    x = np.array(x)
                    y = np.array(y)

                    selected_tag = np.arange(m)
                    np.random.shuffle(selected_tag)
                    selected_tag = selected_tag[:N]

                    x = x[selected_tag]
                    y = y[selected_tag]

            else:  # 大范围地随机搜索
                w = 16
                half_w = w >> 1
                n = 2 * N
                sx, sy = self.random_gen.generate_random(n, x1, x2, y1, y2)
                grad_list = []
                for xx, yy in zip(sx, sy):
                    rr, cc = rectangle((yy - half_w - y0, xx - half_w - x0), extent=(w, w))

                    select_y = (rr >= 0) & (rr < self.valid_area_height)
                    select_x = (cc >= 0) & (cc < self.valid_area_width)
                    select = select_x & select_y
                    max_grad = np.max(sobel_img[rr[select], cc[select]])
                    grad_list.append(max_grad)

                index = np.array(grad_list).argsort()
                sx = sx[index[-N:]]
                sy = sy[index[-N:]]

                x.extend(sx)
                y.extend(sy)

        return list(zip(x, y))

    # 在有效区域内采样，效果不好，因为无效区域内没有值时，插值后会出现误差
    # def get_random_seeds_ex3(self, N, x0, y0, x1, x2, y1, y2, sobel_img, threshold):
    #     '''
    #     获取随机的种子点坐标（扩展算法）
    #     :param N: 获取种子点的数量
    #     :param x0: Sobel_image的左上角在全切片的绝对位置坐标x
    #     :param y0: Sobel_image的左上角在全切片的绝对位置坐标y
    #     :param x1: 在全切片中当前检测区域的左上角x
    #     :param x2: 在全切片中当前检测区域的右下角x
    #     :param y1: 在全切片中当前检测区域的左上角y
    #     :param y2: 在全切片中当前检测区域的右下角y
    #     :param sobel_img: 梯度图
    #     :param threshold: 边界阈值
    #     :return: 种子点
    #     '''
    #     # 目前，自适应采样分为三个阶段进行：
    #     # 1）第一轮，全局的Haltan随机采样，以此生成Sobel图
    #     # 2）第二轮开始，直到检测阈值Threshold达到设定范围之前的阶段：根据WxW区域内梯度局部极大值进行种子点的选择。
    #     #       相当于图像分辨率下降后的随机搜索过程，加速寻找可能性的区域。
    #     # 3）检测阈值达到设定值Threshold后， 算法进入精确的搜索阶段：只选择梯度值越过设定Threshold的点进行采样，
    #     #       直到采样过密，采样点重合条件达到，退出算法。
    #     if sobel_img is None:  # 第一轮
    #         n = N
    #         x, y = self.random_gen.generate_random(n, x1, x2, y1, y2)
    #     else:
    #         x = []
    #         y = []
    #         if threshold > self.search_therhold:  # 精确搜索开始
    #             n = 4 * N
    #             while len(x) <= N/10:
    #                 sx, sy = self.random_gen.generate_random_by_mask(n, x1, x2, y1, y2, self.effi_seeds)
    #
    #                 prob = sobel_img[sy - y0, sx - x0]
    #                 index = prob >= threshold
    #                 sx = sx[index]
    #                 sy = sy[index]
    #
    #                 x.extend(sx)
    #                 y.extend(sy)
    #
    #             half_w = 128
    #
    #             if len(x) < N:
    #                 for xx, yy in zip(sx, sy):
    #                     nx1 = max(xx - half_w, x1)
    #                     nx2 = min(xx + half_w, x2)
    #                     ny1 = max(yy - half_w, y1)
    #                     ny2 = min(yy + half_w, y2)
    #                     sx2, sy2 = self.random_gen.generate_random_by_mask(n, nx1, nx2, ny1, ny2, self.effi_seeds)
    #
    #                     prob = sobel_img[sy2 - y0, sx2 - x0]
    #                     index = prob >= threshold
    #                     sx = sx2[index]
    #                     sy = sy2[index]
    #
    #                     x.extend(sx)
    #                     y.extend(sy)
    #
    #             # 从当前的x和y的结果中只选择N个
    #             m = len(x)
    #             if m > N:
    #                 x = np.array(x)
    #                 y = np.array(y)
    #
    #                 selected_tag = np.arange(m)
    #                 np.random.shuffle(selected_tag)
    #                 selected_tag = selected_tag[:N]
    #
    #                 x = x[selected_tag]
    #                 y = y[selected_tag]
    #
    #         else:  # 大范围地随机搜索
    #             w = 16
    #             half_w = w >> 1
    #             n = 2 * N
    #             sx, sy = self.random_gen.generate_random_by_mask(n, x1, x2, y1, y2, self.effi_seeds)
    #             grad_list = []
    #             for xx, yy in zip(sx, sy):
    #                 rr, cc = rectangle((yy - half_w - y0, xx - half_w - x0), extent=(w, w))
    #
    #                 select_y = (rr >= 0) & (rr < self.valid_area_height)
    #                 select_x = (cc >= 0) & (cc < self.valid_area_width)
    #                 select = select_x & select_y
    #                 max_grad = np.max(sobel_img[rr[select], cc[select]])
    #                 grad_list.append(max_grad)
    #
    #             index = np.array(grad_list).argsort()
    #             sx = sx[index[-N:]]
    #             sy = sy[index[-N:]]
    #
    #             x.extend(sx)
    #             y.extend(sy)
    #
    #     return tuple(zip(x, y))

    # def get_effective_seeds(self, x1, y1, x2, y2, coordinate_scale, seed_scale):
    #     xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * seed_scale / coordinate_scale).astype(np.int)
    #
    #     effi_region =self._imgCone.get_effective_zone(seed_scale)
    #     mask = np.zeros(effi_region.shape, dtype=np.bool)
    #     mask[yy1:yy2, xx1:xx2] = effi_region[yy1:yy2, xx1:xx2]
    #
    #     result = mask.nonzero()
    #
    #     effi_seeds = set()
    #     for xx, yy in zip(result[1], result[0]):
    #         effi_seeds.add((xx, yy))
    #
    #     return effi_seeds