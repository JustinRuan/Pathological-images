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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib

class Locator(object):
    def __init__(self, params):
        self._params = params


    def calcuate_location_features(self, thresh_list, chosen = None):

        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)
        feature_data = {}
        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-14]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz":
                print("loading dataa : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file))
                x1 = result["x1"]
                y1 = result["y1"]
                coordinate_scale = result["scale"]
                assert coordinate_scale==1.25, "Scale is Error!"

                cancer_map = result["cancer_map"]

                points = self.search_local_feature_points(cancer_map, thresh_list, x1, y1)
                feature_data[slice_id] = points

        return feature_data

    def search_local_feature_points(self, cancer_map, thresh_list, x_letftop, y_lefttop):
        '''
        生成用于FROC检测 的数据点
        :param cancer_map:
        :param thresh_list:
        :param x_letftop:
        :param y_lefttop:
        :return:
        '''
        features = {}
        last_thresh = 1.0
        for thresh in thresh_list:
            region = cancer_map > thresh
            candidated_tag = morphology.label(region, neighbors=8, connectivity=2)
            num_tag = np.amax(candidated_tag)
            properties = measure.regionprops(candidated_tag, intensity_image=cancer_map, cache=True)

            for index in range(1, num_tag + 1):
                p = properties[index - 1]
                # 提取特征

                max_value = p.max_intensity
                if max_value < last_thresh and p.major_axis_length > 35:
                    centroid = p.weighted_centroid
                    # 坐标从1.25倍镜下的
                    x = np.rint(centroid[1] + x_letftop).astype(np.int)
                    y = np.rint(centroid[0] + y_lefttop).astype(np.int)

                    f = [p.area, p.extent, p.major_axis_length, p.minor_axis_length,
                         p.max_intensity, p.mean_intensity]
                    f.extend(p.weighted_moments_hu.flatten())

                    features[(x, y)] = f

            last_thresh = thresh
        print("get points: ", len(features))
        return features

    def create_train_data(self, feature_data):
        imgCone = ImageCone(self._params, Open_Slide())
        X = []
        Y = []
        for slice_id, features in feature_data.items():
            print("processing ", slice_id, '... ...')
            # 读取数字全扫描切片图像
            tag = imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                                     'Train_Tumor/%s.xml' % slice_id, slice_id)
            all_mask = imgCone.create_mask_image(self._params.GLOBAL_SCALE, 0)
            mask = all_mask['C']
            for (xx, yy), f in features.items():
                if mask[yy, xx]:
                    Y.append(1)
                else:
                    Y.append(0)
                X.append(f)
        print("count of X :", len(X), "count of cancer points", sum(Y))
        return X, Y

    # def train_svm(self, X, Y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
    #
    #     max_iter = 5000
    #     model_params = [{'C': 0.0001}, {'C': 0.001}, {'C': 0.01}, {'C': 0.1},
    #                     {'C': 0.5}, {'C': 1.0}, {'C': 1.2}, {'C': 1.5},
    #                     {'C': 2.0}, {'C': 10.0}]
    #
    #     result = {'pred': None, 'score': 0, 'clf': None}
    #
    #     feature_num = len(X_train[0])
    #     for params in model_params:
    #         clf = LinearSVC(**params, max_iter=max_iter, verbose=0, class_weight='balanced') # class_weight='balanced'
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    #         score = metrics.accuracy_score(y_test, y_pred)
    #         print('feature num = {}, C={:8f} => score={:5f}'.format(feature_num, params['C'], score))
    #
    #         if score > result["score"]:
    #             result = {'pred': y_pred, 'score': score, 'clf': clf}
    #
    #     print("the best score = {}".format(result["score"]))
    #
    #     target_names = ['normal', 'cancer']
    #     report = metrics.classification_report(y_test, result["pred"], target_names=target_names, digits = 4, output_dict=True)
    #     # print("Classification report for classifier %s:\n%s\n"
    #     #       % (result["clf"], report))
    #     print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, result["pred"]))
    #
    #     model_file = self._params.PROJECT_ROOT + "/models/locator_svm_{:.4f}_{:.4f}.model".format(report['normal']['f1-score'],
    #                                                                                               report['cancer']['f1-score'])
    #     joblib.dump(result["clf"], model_file)
    #     return

    def train_svm(self, X, Y):
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
        X_train, y_train = X, Y

        max_iter = 5000
        model_params = [{'C': 0.0001}, {'C': 0.001}, {'C': 0.01}, {'C': 0.1},
                        {'C': 0.5}, {'C': 1.0}, {'C': 1.2}, {'C': 1.5},
                        {'C': 2.0}, {'C': 10.0}]

        result = {'pred': None, 'score': 0, 'clf': None}

        feature_num = len(X_train[0])
        for params in model_params:
            clf = LinearSVC(**params, max_iter=max_iter, verbose=0, class_weight={0:0.1, 1:10}) # class_weight='balanced'
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            score = metrics.accuracy_score(y_train, y_pred)
            print('feature num = {}, C={:8f} => score={:5f}'.format(feature_num, params['C'], score))

            if score > result["score"]:
                result = {'pred': y_pred, 'score': score, 'clf': clf}

        print("the best score = {}".format(result["score"]))

        target_names = ['normal', 'cancer']
        report = metrics.classification_report(y_train, result["pred"], target_names=target_names, digits = 4, output_dict=True)
        # print("Classification report for classifier %s:\n%s\n"
        #       % (result["clf"], report))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, result["pred"]))

        model_file = self._params.PROJECT_ROOT + "/models/locator_svm_{:.4f}_{:.4f}.model".format(report['normal']['f1-score'],
                                                                                                  report['cancer']['f1-score'])
        joblib.dump(result["clf"], model_file)
        print("saved : ", model_file)
        return

    def evaluate(self, X, Y):
        model_file = self._params.PROJECT_ROOT + "/models/locator_svm_0.7778_0.8491.model"
        clf = joblib.load(model_file)
        pred = clf.decision_function(X)
        for pp, yy in zip(pred, Y):
            print("{:d}, \t{:.4f}".format(yy, pp))

    def output_result_csv(self, thresh_list, chosen = None):
        '''
        生成用于FROC检测用的CSV文件
        :param chosen: 选择哪些切片的结果将都计算，如果为None，则目录下所有的npz对应的结果将被计算
        :return:
        '''
        model_file = self._params.PROJECT_ROOT + "/models/locator_svm_0.7778_0.8491.model"
        clf = joblib.load(model_file)

        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)
        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-14]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz":
                result = np.load("{}/{}".format(save_path, result_file))
                x1 = result["x1"]
                y1 = result["y1"]
                coordinate_scale = result["scale"]
                assert coordinate_scale==1.25, "Scale is Error!"

                cancer_map = result["cancer_map"]
                # print("max :", np.max(cancer_map), "min :", np.min(cancer_map))

                points = self.search_local_feature_points(cancer_map, thresh_list, x1, y1)
                candidated_result = []
                for (xx, yy), f in points.items():
                    # pred = clf.decision_function([f])
                    # if pred > -100:
                        # 坐标从1.25倍镜下变换到40倍镜下
                    candidated_result.append({"x": 32 * xx, "y": 32 * yy, "prob": f[4]})


                csv_filename = "{0}/{1}.csv".format(save_path, slice_id)
                with open(csv_filename, 'w', newline='')as f:
                    f_csv = csv.writer(f)
                    for item in candidated_result:
                        f_csv.writerow([item["prob"], item["x"], item["y"]])

                print("完成 ", slice_id)
        return