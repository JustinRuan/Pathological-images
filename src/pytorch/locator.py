#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-18'

"""

import numpy as np
import os
from skimage import measure
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from sklearn import metrics
from skimage import morphology
import csv
from core import *
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
from skimage.morphology import extrema, local_maxima
from skimage.morphology import square
from sklearn.ensemble import IsolationForest
from pytorch.cancer_map import CancerMapBuilder
from scipy.spatial import distance
from sklearn.cluster import KMeans
from skimage import filters
from pytorch.slide_predictor import BasePredictor
from sklearn.model_selection import GridSearchCV
from pytorch.slide_predictor import SlidePredictor

class Locator(BasePredictor):
    def __init__(self, params):
        super(Locator, self).__init__(params, "Locator")

    # def output_result_csv(self, sub_path, tag, chosen = None):
    #     '''
    #     生成用于FROC检测用的CSV文件
    #     :param chosen: 选择哪些切片的结果将都计算，如果为None，则目录下所有的npz对应的结果将被计算
    #     :return:
    #     '''
    #     project_root = self._params.PROJECT_ROOT
    #     save_path = "{}/results".format(project_root)
    #
    #     if tag == 0:
    #         code = "_history.npz"
    #     else:
    #         code = "_history_v{}.npz".format(tag)
    #
    #     K = len(code)
    #
    #     for result_file in os.listdir(save_path):
    #         ext_name = os.path.splitext(result_file)[1]
    #         slice_id = result_file[:-K]
    #         if chosen is not None and slice_id not in chosen:
    #             continue
    #
    #         if ext_name == ".npz" and code in result_file:
    #             history, label_map, x1, x2, y1, y2 = self.load_history_labelmap(result_file, slice_id)
    #
    #             candidated_result = self.search_local_extremum_points_fusion(history, label_map, x1, y1, x2, y2)
    #             print("count =", len(candidated_result))
    #
    #             csv_filename = "{0}/{1}/{2}.csv".format(save_path,sub_path, slice_id)
    #             with open(csv_filename, 'w', newline='')as f:
    #                 f_csv = csv.writer(f)
    #                 for item in candidated_result:
    #                     f_csv.writerow([item["prob"], item["x"], item["y"]])
    #
    #             print("完成 ", slice_id)
    #     return

    def output_result_csv(self, sub_path, tag, chosen = None):
        '''
        生成用于FROC检测用的CSV文件
        :param chosen: 选择哪些切片的结果将都计算，如果为None，则目录下所有的npz对应的结果将被计算
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        sp = SlidePredictor(self._params)

        if tag == 0:
            code = "_history.npz"
        else:
            code = "_history_v{}.npz".format(tag)

        K = len(code)

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz" and code in result_file:
                history, label_map, x1, x2, y1, y2 = self.load_history_labelmap(result_file, slice_id)
                is_Tumor = sp.predict(history, x1, y1, x2, y2)
                # is_Tumor = True

                candidated_result = []
                if is_Tumor:
                    # candidated_result = self.search_local_extremum_points_fusion(history, label_map, x1, y1, x2, y2)
                    candidated_result = self.search_local_extremum_points_max2(history, x1, y1, x2, y2)

                print("candidated count =", len(candidated_result))

                csv_filename = "{0}/{1}/{2}.csv".format(save_path,sub_path, slice_id)
                with open(csv_filename, 'w', newline='')as f:
                    f_csv = csv.writer(f)
                    for item in candidated_result:
                        f_csv.writerow([item["prob"], item["x"], item["y"]])

                print("完成 ", slice_id)
        return

    def output_result_csv_method2(self, sub_path, tag, chosen = None):
        '''
        生成用于FROC检测用的CSV文件
        :param chosen: 选择哪些切片的结果将都计算，如果为None，则目录下所有的npz对应的结果将被计算
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        model_file = self._params.PROJECT_ROOT + "/models/Locator_svm_0.8536_0.7725.model"

        clf = joblib.load(model_file)

        if tag == 0:
            code = "_history.npz"
        else:
            code = "_history_v{}.npz".format(tag)

        K = len(code)

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz" and code in result_file:
                history, label_map, x1, x2, y1, y2 = self.load_history_labelmap(result_file, slice_id)

                candidated_result = self.search_local_extremum_points_predict(clf, history, label_map, x1, y1, x2, y2)
                print("count =", len(candidated_result))

                csv_filename = "{0}/{1}/{2}.csv".format(save_path,sub_path, slice_id)
                with open(csv_filename, 'w', newline='')as f:
                    f_csv = csv.writer(f)
                    for item in candidated_result:
                        f_csv.writerow([item["prob"], item["x"], item["y"]])

                print("完成 ", slice_id)
        return

    def load_history_labelmap(self, result_file, slice_id):
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        print("loading data : {}, {}".format(slice_id, result_file))
        result = np.load("{}/{}".format(save_path, result_file), allow_pickle=True)
        x1 = result["x1"]
        y1 = result["y1"]
        x2 = result["x2"]
        y2 = result["y2"]
        coordinate_scale = result["scale"]
        assert coordinate_scale == 1.25, "Scale is Error!"
        history = result["history"].item()

        labelmap_filename = "{}/data/Segmentations/{}_labelmap.npz".format(project_root, slice_id)
        print("loading superpixels : {}, {}".format(slice_id, labelmap_filename))
        result = np.load(labelmap_filename, allow_pickle=True)
        bx1 = result["x1"]
        by1 = result["y1"]
        bx2 = result["x2"]
        by2 = result["y2"]
        coordinate_scale = result["scale"]
        assert coordinate_scale == 1.25, "Scale is Error!"
        assert x1 == bx1 and y1 == by1 and x2 == bx2 and y2 == by2, "coordinate is Error!"
        label_map = result["labelmap"]
        return history, label_map, x1, x2, y1, y2

    def load_true_mask(self, slice_id, x1, y1, x2, y2):
        mask_path = "{}/data/true_masks".format(self._params.PROJECT_ROOT)

        mask_filename = "{}/{}_true_mask.npy".format(mask_path, slice_id)
        if os.path.exists(mask_filename):
            mask_img = np.load(mask_filename)
            mask_img = mask_img[y1:y2, x1:x2]
        else:
            h = y2 - y1
            w = x2 - x1
            mask_img = np.zeros((h, w), dtype=np.bool)

        return mask_img

    # # 模式1：均匀的选择
    # def search_local_extremum_points(self, history, x_letftop, y_lefttop):
    #     low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, t=-0.5)
    #     # low_prob_thresh = 0.5
    #     candidated = []
    #     for (x, y), f in history.items():
    #         prob = 1 / (1 + np.exp(-f))
    #         if low_prob_thresh < prob:
    #             xx = x + x_letftop
    #             yy = y + y_lefttop
    #             candidated.append((prob, xx, yy))
    #
    #     candidated.sort(key=lambda x: (x[0]), reverse=True)
    #
    #     resolution = 0.243
    #     level = 5
    #     Threshold = 5 * 75 / (resolution * pow(2, level) * 2)
    #     result = []
    #     while len(candidated) > 0:
    #         m = candidated.pop(0)
    #         mx = m[1]
    #         my = m[2]
    #         mprob = m[0]
    #         tx = [mx]
    #         ty = [my]
    #         temp = []
    #         for prob, x, y in candidated:
    #             dist = distance.euclidean([mx, my], [x, y])
    #             if dist > Threshold or mprob - prob > 0.15:
    #                 temp.append((prob, x, y))
    #             else:
    #                 tx.append(x)
    #                 ty.append(y)
    #
    #         mx = np.rint(np.mean(np.array(tx))).astype(np.int)
    #         my = np.rint(np.mean(np.array(ty))).astype(np.int)
    #
    #         result.append((mprob, mx, my))
    #         candidated = temp
    #
    #     candidated = []
    #     for prob, x, y in result:
    #         candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})
    #
    #     return candidated
    #
    # 模式2：局部极值输出
    def search_local_extremum_points_max(self, history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):
        low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, t = -0.5)

        cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
                                                    self._params.GLOBAL_SCALE)

        h = 0.02
        h_maxima = extrema.h_maxima(cancer_map, h, selem=square(7))
        xy = np.nonzero(h_maxima)

        candidated = []
        for y, x in zip(xy[0], xy[1]):
            prob = cancer_map[y, x]
            if prob > low_prob_thresh:
                x = x + x_letftop
                y = y + y_lefttop
                candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})

        return candidated

    def search_local_extremum_points_max2(self, history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):
        # for train set: t,c,h = -0.5, 50, 0.02
        t_thresh = 0  # -0.5
        c_thresh = 10  # 50
        h_param = 0.02 # 0.02
        low_prob_thresh, high_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, t = t_thresh)
        print("low_prob_thresh = {:.4f}, high_prob_thresh = {:.4f}".format(low_prob_thresh, high_prob_thresh))

        cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
                                                    self._params.GLOBAL_SCALE)

        h = h_param
        h_maxima = extrema.h_maxima(cancer_map, h, selem=square(7))
        xy = np.nonzero(h_maxima)

        sorted_points = []
        for y, x in zip(xy[0], xy[1]):
            prob = cancer_map[y, x]
            if prob > low_prob_thresh:
                sorted_points.append((prob, x, y))
        sorted_points.sort(key=lambda x: (x[0]), reverse=True)

        resolution = 0.243
        level = 5
        Threshold = 3 * 75 / (resolution * pow(2, level) * 2)
        result = []
        while len(sorted_points) > 0:
            m = sorted_points.pop(0)
            mx = m[1]
            my = m[2]
            mprob = m[0]
            tx = [mx]
            ty = [my]
            temp = []
            for prob, x, y in sorted_points:
                dist = distance.euclidean([mx, my], [x, y])
                if dist > Threshold or mprob - prob > 0.15:
                    temp.append((prob, x, y))
                else:
                    tx.append(x)
                    ty.append(y)

            mx = np.rint(np.mean(np.array(tx))).astype(np.int)
            my = np.rint(np.mean(np.array(ty))).astype(np.int)

            result.append((mprob, mx, my))
            sorted_points = temp

        candidated = []
        count = 0
        for prob, x, y in result:
            if prob > high_prob_thresh or (prob > low_prob_thresh and count < c_thresh):
                x = x + x_letftop
                y = y + y_lefttop
                candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})
                count += 1

        return candidated

    # 模式3
    def search_local_extremum_points2(self, history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):
        low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history)
        # low_prob_thresh = 0.5
        value = 1 / (1 + np.exp(-np.array(list(history.values()))))
        positive_part = np.reshape(value[value > low_prob_thresh], (-1, 1))
        round_value = np.unique(np.round(positive_part, decimals=2))
        thresh_list = np.sort(round_value)[::-1]

        cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
                                                    self._params.GLOBAL_SCALE)

        candidated = []
        last_thresh = 1.0
        for thresh in thresh_list:
            region = cancer_map >= thresh
            candidated_tag = morphology.label(region, neighbors=8, connectivity=2)
            num_tag = np.amax(candidated_tag)
            properties = measure.regionprops(candidated_tag, intensity_image=cancer_map, cache=True, coordinates='rc')

            for index in range(1, num_tag + 1):
                p = properties[index - 1]
                # 提取特征
                max_value = p.max_intensity
                if max_value < last_thresh:
                    select_region = candidated_tag == index
                    new_map = np.zeros(cancer_map.shape)
                    new_map[select_region] = cancer_map[select_region]

                    pos = np.nonzero(new_map == max_value)
                    max_count = len(pos[0])
                    k = max_count // 2
                    # 坐标从1.25倍镜下的
                    x = pos[1][k] + x_letftop
                    y = pos[0][k] + y_lefttop

                    candidated.append((p.max_intensity, x, y))

            last_thresh = thresh

        candidated.sort(key=lambda x: (x[0]), reverse=True)

        resolution = 0.243
        level = 5
        Threshold = 5 * 75 / (resolution * pow(2, level) * 2)
        result = []
        while len(candidated) > 0:
            m = candidated.pop(0)
            mx = m[1]
            my = m[2]
            mprob = m[0]
            tx = [mx]
            ty = [my]
            temp = []
            for prob, x, y in candidated:
                dist = distance.euclidean([mx, my], [x, y])
                if dist > Threshold or mprob - prob > 0.15:
                    temp.append((prob, x, y))
                else:
                    tx.append(x)
                    ty.append(y)

            mx = np.rint(np.mean(np.array(tx))).astype(np.int)
            my = np.rint(np.mean(np.array(ty))).astype(np.int)

            result.append((mprob, mx, my))
            candidated = temp

        candidated = []
        for prob, x, y in result:
            candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})

        return candidated

    # 模式4：
    def search_local_extremum_points_fusion(self, history, label_map, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):

        c_thresh = 50

        low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, 0.5)

        cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
                                                    self._params.GLOBAL_SCALE)

        h = 0.02
        h_maxima = extrema.h_maxima(cancer_map, h, selem=square(7))
        xy = np.nonzero(h_maxima)

        sobel_img = filters.sobel(cancer_map)
        feat_thresh = 0.5  # feat = -1对应概率0.27, feat = -0.5 对应0.38，feat = -0.2 对应0.45
        grad_thresh = 0.05
        f_properties = measure.regionprops(label_map, intensity_image=cancer_map, cache=True, coordinates='rc')
        g_properties = measure.regionprops(label_map, intensity_image=sobel_img, cache=True, coordinates='rc')

        candidated_region = {}
        regions_extres = {}
        for y, x in zip(xy[0], xy[1]):
            region_id = label_map[y, x]
            if region_id not in regions_extres.keys():
                regions_extres[region_id] = (x, y)
            else:
                new_prob = cancer_map[y, x]
                old_x, old_y = regions_extres[region_id]
                old_prob = cancer_map[old_y, old_x]
                if new_prob > old_prob:
                    regions_extres[region_id] = (x, y)

        max_label = np.amax(label_map)
        print("max label =", max_label,)

        for index in range(1, max_label + 1):
            if index in regions_extres.keys():
                x, y = regions_extres[index]
                candidated_region[index] = (x, y)
            else:
                p = f_properties[index - 1]
                g = g_properties[index - 1]
                max_prob = p.max_intensity
                max_grad = g.max_intensity
                assert p.label == index, "Error: p.label != index"
                if max_prob >= feat_thresh and max_grad >= grad_thresh:
                    y, x = p.weighted_centroid
                    if not (np.isnan(x) or np.isnan(y)):
                        candidated_region[index] = (int(x), int(y))

        sorted_points = []
        for x, y in candidated_region.values():
            prob = cancer_map[y, x]
            if prob > low_prob_thresh:
                sorted_points.append((prob, x, y))
        sorted_points.sort(key=lambda x: (x[0]), reverse=True)

        resolution = 0.243
        level = 5
        Threshold = 3 * 75 / (resolution * pow(2, level) * 2)
        result = []
        while len(sorted_points) > 0:
            m = sorted_points.pop(0)
            mx = m[1]
            my = m[2]
            mprob = m[0]
            tx = [mx]
            ty = [my]
            temp = []
            for prob, x, y in sorted_points:
                dist = distance.euclidean([mx, my], [x, y])
                if dist > Threshold or mprob - prob > 0.15:
                    temp.append((prob, x, y))
                else:
                    tx.append(x)
                    ty.append(y)

            mx = np.rint(np.mean(np.array(tx))).astype(np.int)
            my = np.rint(np.mean(np.array(ty))).astype(np.int)

            result.append((mprob, mx, my))
            sorted_points = temp

        candidated = []
        count = 0
        for prob, x, y in result:
            if count < c_thresh:
                x = x + x_letftop
                y = y + y_lefttop
                candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})
                count += 1


        # candidated = []
        # for x, y in candidated_region.values():
        #     prob = cancer_map[y, x]
        #     if prob > low_prob_thresh:
        #         x = x + x_letftop
        #         y = y + y_lefttop
        #         candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})

        return candidated

    # # 模式4：
    # def search_local_extremum_points_fusion(self, history, label_map, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):
    #     # low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, 0)
    #     low_prob_thresh = 0.3
    #     high_prob_thresh = 0.4
    #     cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
    #     cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
    #                                                 self._params.GLOBAL_SCALE)
    #
    #     h = 0.02
    #     h_maxima = extrema.h_maxima(cancer_map, h, selem=square(7))
    #     xy = np.nonzero(h_maxima)
    #
    #     # sobel_img = filters.sobel(cancer_map)
    #     # h_maxima2 = extrema.h_minima(sobel_img, h, selem=square(7))
    #     # xy2 = np.nonzero(h_maxima2)
    #     ex = xy[1]
    #     ey = xy[0]
    #     # ex = np.append(xy[1], xy2[1])
    #     # ey = np.append(xy[0], xy2[0])
    #     # # feat_thresh = -1  # feat = -1对应概率0.27, feat = -0.5 对应0.38，feat = -0.2 对应0.45
    #     #
    #     # grad_thresh = 0.01
    #     # f_properties = measure.regionprops(label_map, intensity_image=cancer_map, cache=True, coordinates='rc')
    #     # g_properties = measure.regionprops(label_map, intensity_image=sobel_img, cache=True, coordinates='rc')
    #
    #     candidated_region = {}
    #     regions_extres = {}
    #     for y, x in zip(ey, ex):
    #         prob = cancer_map[y, x]
    #         region_id = label_map[y, x]
    #         if region_id not in regions_extres.keys():
    #             regions_extres[region_id] = (prob, x, y)
    #         else:
    #             new_prob = cancer_map[y, x]
    #             old_prob, old_x, old_y = regions_extres[region_id]
    #             if new_prob > old_prob:
    #                 regions_extres[region_id] = (new_prob, x, y)
    #
    #     max_label = np.amax(label_map)
    #     print("max label =", max_label,)
    #
    #     for index in range(1, max_label + 1):
    #         if index in regions_extres.keys():
    #             prob, x, y = regions_extres[index]
    #             candidated_region[index] = (prob, x, y)
    #         # else:
    #         #     p = f_properties[index - 1]
    #         #     g = g_properties[index - 1]
    #         #     max_prob = p.max_intensity
    #         #     max_grad = g.max_intensity
    #         #     assert p.label == index, "Error: p.label != index"
    #         #     if max_prob >= low_prob_thresh and max_grad >= grad_thresh:
    #         #         tag = np.logical_and(label_map == index, cancer_map == max_prob)
    #         #         pos = np.nonzero(tag)
    #         #         max_count = len(pos[0])
    #         #         k = max_count // 2
    #         #         # 坐标从1.25倍镜下的
    #         #         x = pos[1][k]
    #         #         y = pos[0][k]
    #         #         candidated_region[index] = (max_prob, int(x), int(y))
    #
    #     sort_regions = sorted(candidated_region.values(), key=lambda s:s[0], reverse=True)
    #
    #     candidated = []
    #     count = 0
    #     for prob, x, y in sort_regions:
    #         if prob > high_prob_thresh or (count < 20 and prob > low_prob_thresh):
    #             x = x + x_letftop
    #             y = y + y_lefttop
    #             candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})
    #             count += 1
    #
    #     return candidated

    # 模式6：
    def search_local_extremum_points_predict(self, clf, history, label_map, x_letftop, y_lefttop, x_rightbottom, y_rightbottom):
        low_prob_thresh = CancerMapBuilder.calc_probability_threshold(history, 0)
        # low_prob_thresh = 0.5
        cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
        cancer_map = cmb.generating_probability_map(history, x_letftop, y_lefttop, x_rightbottom, y_rightbottom,
                                                    self._params.GLOBAL_SCALE)

        h = 0.1
        h_maxima = extrema.h_maxima(cancer_map, h, selem=square(7))
        xy = np.nonzero(h_maxima)

        regions_prbability = self.calc_regions_prbability(clf, history, label_map)

        p_properties = measure.regionprops(label_map, intensity_image=cancer_map, cache=True, coordinates='rc')

        candidated_region = {}
        regions_extres = {}
        for y, x in zip(xy[0], xy[1]):
            region_id = label_map[y, x]
            if region_id not in regions_extres.keys():
                regions_extres[region_id] = (x, y)
            else:
                new_prob = cancer_map[y, x]
                old_x, old_y = regions_extres[region_id]
                old_prob = cancer_map[old_y, old_x]
                if new_prob > old_prob:
                    regions_extres[region_id] = (x, y)

        max_label = np.amax(label_map)
        print("max label =", max_label,)

        for index in range(1, max_label + 1):
            if index in regions_prbability.keys() and\
                    regions_prbability[index] >= 0.3:
                if index in regions_extres.keys():
                    x, y = regions_extres[index]
                    candidated_region[index] = (x, y)
                else:
                    p = p_properties[index - 1]
                    y, x = p.weighted_centroid
                    if not (np.isnan(x) or np.isnan(y)):
                        candidated_region[index] = (int(x), int(y))

        candidated = []
        for x, y in candidated_region.values():
            prob = cancer_map[y, x]
            if prob >= low_prob_thresh:
                x = x + x_letftop
                y = y + y_lefttop
                candidated.append({"x": 32 * x, "y": 32 * y, "prob": prob})

        return candidated

    def calc_regions_prbability(self, clf, history, label_map):
        features = self.calc_superpixel_feature(history, label_map)
        result = {}
        for index, feature in features.items():
            y_pred = clf.predict_proba([feature])
            prob = y_pred[:, 1]
            result[index] = prob
        return result

    def calc_superpixel_feature(self,history, label_map, cancer_map):
        regions_probs = {}
        mode = 2
        if mode == 1: # by cancer map
            selem_size = 8
            cancer_map = morphology.erosion(cancer_map,square(selem_size))
            max_label = np.amax(label_map)
            for index in range(1, max_label + 1):
                tag = label_map == index
                data = cancer_map[tag].ravel()
                regions_probs[index] = data
        elif mode == 2: # by history
            for (x, y), f in history.items():
                region_id = label_map[y, x]
                prob = 1 / (1 + np.exp(-f))
                if region_id not in regions_probs.keys():
                    regions_probs[region_id] = [prob]
                else:
                    regions_probs[region_id].append(prob)

        DIM = 5
        regions_features = {}
        for id, data in regions_probs.items():
            data = np.array(data)
            data = data[data >= 0.5]
            L = len(data)
            if L > 0:
                max_prob = np.max(data)
                hist = np.histogram(data, bins=DIM - 2, range=(0.5, 1), density=False)
                feature = np.array(hist[0], dtype=np.float) / L
                feature = np.append(feature, [L, max_prob])
                regions_features[id] = feature
            else:
                regions_features[id] = np.zeros((DIM,))

        return regions_features

    def calc_superpixel_label(self, true_mask, label_map):
        mask = np.array(true_mask, dtype=np.int)
        L_properties = measure.regionprops(label_map, intensity_image=mask, cache=True, coordinates='rc')
        max_label = np.amax(label_map)
        print("max label =", max_label,)

        threshold = 0.5
        regions_label = {}
        for index in range(1, max_label + 1):
            sub_label = L_properties[index - 1]
            assert sub_label.label == index, "Error: label is not match!"
            if sub_label.mean_intensity > threshold:
                regions_label[index] = 1
            else:
                regions_label[index] = 0

        return regions_label

    def create_train_data(self, tag, slide_idset):
        project_root = self._params.PROJECT_ROOT

        if tag == 0:
            code = "_history.npz"
        else:
            code = "_history_v{}.npz".format(tag)

        X, Y = [], []
        tempXY = []
        for slide_id in slide_idset:
            print(slide_id, "begin ...")
            result_file = "{}{}".format(slide_id, code)
            history, label_map, x1, x2, y1, y2 = self.load_history_labelmap(result_file, slide_id)
            true_mask = self.load_true_mask(slide_id, x1, y1, x2, y2)

            regions_features = self.calc_superpixel_feature(history, label_map)
            regions_label = self.calc_superpixel_label(true_mask,label_map)

            for region_id, features in regions_features.items():
                if region_id in regions_label.keys():
                    label = regions_label[region_id]
                    m = np.append(features, [label]).tolist()
                    if len(tempXY) == 0 or (m not in tempXY):
                        X.append(features)
                        Y.append(label)
                        tempXY.append(m)

                else:
                    print(slide_id, "not in regions_label!")

        return X, Y

    def train_svm(self,file_name, test_name):
        X_test, X_train, y_test, y_train = self.read_data(file_name, test_name, mode=3)

        max_iter = 5000

        #'kernel': ('linear', 'poly', 'rbf'),
        # parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.2, 1.5, 2.0, 10]}
        parameters = {'C': [ 1, ]}
        # parameters = { 'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.2, 1.5, 2.0, 10],
        #               'gamma':[2, 1, 0.1, 0.5, 0.01, 0.001, 0.0001]}

        svc = svm.SVC(kernel='rbf', probability=True, max_iter=max_iter,verbose=0,)
        grid  = GridSearchCV(svc, parameters, cv=3)
        grid .fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        clf = grid.best_estimator_
        self.save_model(clf, "linearsvm", X_test, y_test)

        return






