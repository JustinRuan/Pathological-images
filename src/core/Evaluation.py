#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-12'

"""

import numpy as np
import os
from skimage import measure
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from sklearn import metrics
from skimage import morphology
import csv

class Evaluation(object):
    def __init__(self, params):
        self._params = params

###################################################################################################
##############   单个切片的ROC检测
###################################################################################################
    @staticmethod
    def calculate_dice_coef(y_true, y_pred):
        smooth = 1.
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)
        return dice

    @staticmethod
    def evaluate_slice_map(cancer_map, true_mask, levels):
        '''
        癌变概率矩阵进行阈值分割后，与人工标记真值进行 评估
        :param threshold: 分割的阈值
        :param cancer_map: 癌变概率矩阵
        :param true_mask: 人工标记真值
        :return: ROC曲线
        '''
        mask_tag = np.array(true_mask).ravel()

        dice_result = []
        for threshold in levels:
            cancer_tag = np.array(cancer_map > threshold).ravel()
            predicted_tags = cancer_tag >= threshold
            dice = Evaluation.calculate_dice_coef(mask_tag, cancer_tag)

            print("Threshold = {:.3f}, Classification report for classifier: \n{}".format(threshold,
                                            metrics.classification_report(mask_tag, predicted_tags, digits=4)))
            print("############################################################")
            # print("Confusion matrix:\n%s" % metrics.confusion_matrix(mask_tag, predicted_tags))

            dice_result.append((threshold, dice))

        for t, value in dice_result:
            print("threshold = {:.3f}, dice coef = {:.6f}".format(t, value))
        print("############################################################")
        # 计算ROC曲线
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(mask_tag, np.array(cancer_map).ravel())
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        print("\n ROC auc: %s" % roc_auc)
        return false_positive_rate, true_positive_rate, roc_auc, dice_result



    def save_result_xml(self, slice_id, x1, y1, coordinate_scale, cancer_map, threshold_list,):
        '''
        生成在ASAP中能显示的癌变概率的等高线
        :param slice_id: 切片编号
        :param x1: 概率图的左上角在全切片的坐标系中的坐标
        :param y1: 左上角的坐标
        :param coordinate_scale: x1，y1的值所应的倍镜数
        :param cancer_map: 概率图
        :param threshold_list: 生成等高线的分割阈值的列表
        :return:
        '''
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        scale = int(40 / GLOBAL_SCALE)
        scale2 = int(40 / coordinate_scale)

        contours_set = {}
        for threshold in threshold_list:
            cancer_tag = np.array(cancer_map > threshold)
            contours = measure.find_contours(cancer_tag, 0.5)

            contours_x40 = []
            for n, contour in enumerate(contours):
                contour = approximate_polygon(np.array(contour), tolerance=0.01)
                c = scale * np.array(contour)  + np.array([y1, x1]) * scale2
                contours_x40.append(c)

            contours_set[threshold] = contours_x40

        self.write_xml(contours_set, slice_id)

    def write_xml(self, contours_set, slice_id):
        '''
        将等高线进行保存
        :param contours_set: 等高线的集合
        :param slice_id: 切片编号
        :return:
        '''
        PROJECT_ROOT = self._params.PROJECT_ROOT
        from xml.dom import minidom
        doc = minidom.Document()
        rootNode = doc.createElement("ASAP_Annotations")
        doc.appendChild(rootNode)

        AnnotationsNode = doc.createElement("Annotations")
        rootNode.appendChild(AnnotationsNode)

        colors = ["#00BB00", "#00FF00", "#FFFF00", "#BB0000", "#FF0000"]
        for k, (key, contours) in enumerate(contours_set.items()):
            Code = "{:.2f}".format(key)
            for i, contour in enumerate(contours):
                # one contour
                AnnotationNode = doc.createElement("Annotation")
                AnnotationNode.setAttribute("Name", str(i))
                AnnotationNode.setAttribute("Type", "Polygon")
                AnnotationNode.setAttribute("PartOfGroup", Code)
                AnnotationNode.setAttribute("Color", colors[k])
                AnnotationsNode.appendChild(AnnotationNode)

                CoordinatesNode = doc.createElement("Coordinates")
                AnnotationNode.appendChild(CoordinatesNode)

                for n, (y, x) in enumerate(contour):
                    CoordinateNode = doc.createElement("Coordinate")
                    CoordinateNode.setAttribute("Order", str(n))
                    CoordinateNode.setAttribute("X", str(x))
                    CoordinateNode.setAttribute("Y", str(y))
                    CoordinatesNode.appendChild(CoordinateNode)

        AnnotationGroups_Node = doc.createElement("AnnotationGroups")
        rootNode.appendChild(AnnotationGroups_Node)

        for k, (key, _) in enumerate(contours_set.items()):
            Code = "{:.2f}".format(key)
            GroupNode = doc.createElement("Group")
            GroupNode.setAttribute("Name", Code)
            GroupNode.setAttribute("PartOfGroup", "None")
            GroupNode.setAttribute("Color", colors[k])
            AnnotationGroups_Node.appendChild(GroupNode)

        f = open("{}/results/{}_output.xml".format(PROJECT_ROOT, slice_id), "w")
        doc.writexml(f, encoding="utf-8")
        f.close()



###################################################################################################
##############   多个切片的FROC检测的计算过程
###################################################################################################










