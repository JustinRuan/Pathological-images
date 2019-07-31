#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-12'

"""

import numpy as np
import os
import sys
from skimage import measure
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from sklearn import metrics
from skimage import morphology
import csv
from core import *
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage.morphology import square
from pytorch.cancer_map import CancerMapBuilder

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

    def calculate_ROC(self, slice_dirname, chosen):

        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        imgCone = ImageCone(self._params, Open_Slide())
        result_auc = []
        K = len("_history.npz")

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz":
                print("loading data : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file))
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()

                cmb = CancerMapBuilder(self._params, x1, y1, x2, y2, 1.25)
                cancer_map = cmb.generating_probability_map(history, extract_scale=40, patch_size=256)

                imgCone.open_slide("{}/{}.tif".format(slice_dirname, slice_id),
                                   '{}/{}.xml'.format(slice_dirname, slice_id), slice_id)

                mask_img = imgCone.create_mask_image(self._params.GLOBAL_SCALE, 0)
                mask_img = mask_img['C']

                h, w = cancer_map.shape
                mask_img = mask_img[y1:y1+h, x1:x1+w]
                area = np.sum(mask_img)
                _, count = morphology.label(mask_img, neighbors=8, connectivity=2, return_num=True)

                false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(mask_img.ravel(),
                                                                                        cancer_map.ravel())
                roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

                dice = Evaluation.calculate_dice_coef(mask_img, cancer_map > 0.5)

                temp = "{}\t{}\t{}\t{}\t{}".format(slice_id, area, count, dice, roc_auc)
                result_auc.append(temp)
                print(temp)

        print("############################################")
        for item in result_auc:
            print(item)

        return

###################################################################################################
##############   多个切片的FROC检测的计算过程
###################################################################################################
    def create_true_mask_file(self, slice_dirname, slice_list, ):
        imgCone = ImageCone(self._params, Open_Slide())
        mask_path = "{}/data/true_masks".format(self._params.PROJECT_ROOT)

        for slice_id in slice_list:
            if "Tumor" in slice_dirname:
                code = "Tumor_{:0>3d}".format(slice_id)
            else:
                code = "test_{:0>3d}".format(slice_id)

            imgCone.open_slide("{}/{}.tif".format(slice_dirname, code),
                               '{}/{}.xml'.format(slice_dirname, code), code)

            mask_img = imgCone.create_mask_image(self._params.GLOBAL_SCALE, 0)
            mask_img = mask_img['C']
            np.save("{}/{}_true_mask.npy".format(mask_path, code), mask_img, allow_pickle=True)
        return

    def evaluation_FROC(self, mask_folder, result_folder):

        result_file_list = []
        result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]

        # level 0对应40倍镜，level 1对应20倍，这里level 5就是对应1.25倍镜下
        EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
        L0_RESOLUTION = 0.243  # pixel resolution at level 0

        FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
        FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
        detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)

        caseNum = 0
        for case in result_file_list:
            print('Evaluating Performance on image:', case[0:-4])
            sys.stdout.flush()
            csvDIR = os.path.join(result_folder, case)
            Probs, Xcorr, Ycorr = self.readCSVContent(csvDIR)

            is_tumor = case[0:5] == 'Tumor'
            if (is_tumor):
                # maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
                maskDIR = "{}/{}_true_mask.npy".format(mask_folder, case[0:-4])
                evaluation_mask = self.computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                ITC_labels = self.computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            else:
                evaluation_mask = 0
                ITC_labels = []

            FROC_data[0][caseNum] = case
            FP_summary[0][caseNum] = case
            detection_summary[0][caseNum] = case
            FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], \
            detection_summary[1][caseNum], FP_summary[1][caseNum] = \
                self.compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
            caseNum += 1

        # Compute FROC curve
        total_FPs, total_sensitivity = self.computeFROC(FROC_data)

        # plot FROC curve
        self.plotFROC(total_FPs, total_sensitivity)
        return

    def computeEvaluationMask(self, mask_file, resolution, level):
        """Computes the evaluation mask.

        Args:
            maskDIR:    the directory of the ground truth mask
            resolution: Pixel resolution of the image at level 0
            level:      The level at which the evaluation mask is made

        Returns:
            evaluation_mask
        """
        pixelarray = np.load(mask_file)
        Threshold = 75/(resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells

        distance = nd.distance_transform_edt(255 * (1 - pixelarray))
        binary = distance < Threshold
        filled_image = nd.morphology.binary_fill_holes(binary)

        evaluation_mask = measure.label(filled_image, connectivity=2)

        return evaluation_mask

    def computeITCList(self, evaluation_mask, resolution, level):
        """Compute the list of labels containing Isolated Tumor Cells (ITC)

        Description:
            A region is considered ITC if its longest diameter is below 200µm.
            As we expanded the annotations by 75µm, the major axis of the object
            should be less than 275µm to be considered as ITC (Each pixel is
            0.243µm*0.243µm in level 0). Therefore the major axis of the object
            in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

        Args:
            evaluation_mask:    The evaluation mask
            resolution:         Pixel resolution of the image at level 0
            level:              The level at which the evaluation mask was made

        Returns:
            Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        """
        max_label = np.amax(evaluation_mask)
        properties = measure.regionprops(evaluation_mask, coordinates='rc')
        Isolated_Tumor_Cells = []
        threshold = 1.0 * 275/ (resolution * pow(2, level))
        for i in range(0, max_label):
            if properties[i].major_axis_length < threshold:
                Isolated_Tumor_Cells.append(i + 1)
        return Isolated_Tumor_Cells

    def readCSVContent(self, csvDIR):
        """Reads the data inside CSV file

        Args:
            csvDIR:    The directory including all the .csv files containing the results.
            Note that the CSV files should have the same name as the original image

        Returns:
            Probs:      list of the Probabilities of the detected lesions
            Xcorr:      list of X-coordinates of the lesions
            Ycorr:      list of Y-coordinates of the lesions
        """
        Xcorr, Ycorr, Probs = ([] for i in range(3))
        csv_lines = open(csvDIR, "r").readlines()
        for i in range(len(csv_lines)):
            line = csv_lines[i]
            elems = line.rstrip().split(',')
            Probs.append(float(elems[0]))
            Xcorr.append(int(elems[1]))
            Ycorr.append(int(elems[2]))
        return Probs, Xcorr, Ycorr

    def compute_FP_TP_Probs(self, Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
        """Generates true positive and false positive stats for the analyzed image

        Args:
            Probs:      list of the Probabilities of the detected lesions
            Xcorr:      list of X-coordinates of the lesions
            Ycorr:      list of Y-coordinates of the lesions
            is_tumor:   A boolean variable which is one when the case cotains tumor
            evaluation_mask:    The evaluation mask
            Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
            level:      The level at which the evaluation mask was made

        Returns:
            FP_probs:   A list containing the probabilities of the false positive detections

            TP_probs:   A list containing the probabilities of the True positive detections

            NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

            detection_summary:   A python dictionary object with keys that are the labels
            of the lesions that should be detected (non-ITC tumors) and values
            that contain detection details [confidence score, X-coordinate, Y-coordinate].
            Lesions that are missed by the algorithm have an empty value.

            FP_summary:   A python dictionary object with keys that represent the
            false positive finding number and values that contain detection
            details [confidence score, X-coordinate, Y-coordinate].
        """

        max_label = np.amax(evaluation_mask)
        FP_probs = []
        TP_probs = np.zeros((max_label,), dtype=np.float32)
        detection_summary = {}
        FP_summary = {}
        for i in range(1, max_label + 1):
            if i not in Isolated_Tumor_Cells:
                label = 'Label ' + str(i)
                detection_summary[label] = []

        FP_counter = 0
        if (is_tumor):
            for i in range(0, len(Xcorr)):
                HittedLabel = evaluation_mask[Ycorr[i] // pow(2, level), Xcorr[i] // pow(2, level)]
                if HittedLabel == 0:
                    FP_probs.append(Probs[i])
                    key = 'FP ' + str(FP_counter)
                    FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                    FP_counter += 1
                elif HittedLabel not in Isolated_Tumor_Cells:
                    if (Probs[i] > TP_probs[HittedLabel - 1]):
                        label = 'Label ' + str(HittedLabel)
                        detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                        TP_probs[HittedLabel - 1] = Probs[i]
        else:
            for i in range(0, len(Xcorr)):
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1

        num_of_tumors = max_label - len(Isolated_Tumor_Cells);
        return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary

    def computeFROC(self, FROC_data):
        """Generates the data required for plotting the FROC curve

        Args:
            FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

        Returns:
            total_FPs:      A list containing the average number of false positives
            per image for different thresholds

            total_sensitivity:  A list containig overall sensitivity of the system
            for different thresholds
        """

        unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
        unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

        total_FPs, total_TPs = [], []
        all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
        for Thresh in all_probs[1:]:
            total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
            total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
        total_FPs.append(0)
        total_TPs.append(0)
        total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
        total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
        return total_FPs, total_sensitivity

    def plotFROC(self, total_FPs, total_sensitivity):
        """Plots the FROC curve

        Args:
            total_FPs:      A list containing the average number of false positives
            per image for different thresholds

            total_sensitivity:  A list containig overall sensitivity of the system
            for different thresholds

        Returns:
            -
        """
        fig = plt.figure()
        plt.xlabel('Average Number of False Positives', fontsize=12)
        plt.ylabel('Metastasis detection sensitivity', fontsize=12)
        fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
        plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
        plt.show()








