#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
from pytorch.detector import Detector, AdaptiveDetector
import numpy as np
from skimage.segmentation import mark_boundaries
from pytorch.cancer_map import CancerMapBuilder

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_detector(unittest.TestCase):

    def test_search_region(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        slice_id = "001"
        # 读取数字全扫描切片图像
        # tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % id,
        #                          'Tumor/tumor_%s.xml' % id, "Tumor_%s" % id)
        tag = imgCone.open_slide("Testing/images/test_%s.tif" % slice_id,
                                 'Testing/images/test_%s.xml' % slice_id, "test_%s" % slice_id)

        detector = Detector(c, imgCone)
        print("y ", detector.ImageHeight, ", x ", detector.ImageWidth)

        x1 = 800
        y1 = 1600
        x2 = 1600
        y2 = 2300
        # ("001", 800, 1600, 1600, 2300),
        test_set = [("001", 100, 100, 2600, 2700),
                    ("016", 0, 200, 3250, 2900),
                    ("021", 0, 2400, 3000, 6500),]

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        effi_seeds = detector.get_effective_seeds(x1, y1, x2, y2, 1.25, 1.25)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
        ax = axes.ravel()

        ax[0].imshow(src_img)
        shape = src_img.shape
        ax[0].set_title("src_img {} x {}".format(shape[0], shape[1]))

        ax[1].imshow(mask_img)
        ax[1].set_title("mask_img")

        for a in ax.ravel():
            a.axis('off')

        plt.show()

        return

    def test_detector(self):
        test_set = [("001", 2100, 3800, 2400, 4000),
                    ("003", 2400, 4700, 2600, 4850)]
        id = 1
        roi = test_set[id]
        slice_id = roi[0]
        x1 = roi[1]
        y1 = roi[2]
        x2 = roi[3]
        y2 = roi[4]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_%s.tif" % slice_id,
                                 'Train_Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        cancer_map, cancer_map2, cancer_map3, cancer_map4 = detector.process(x1, y1, x2, y2, 1.25, interval=64)

        # show result
        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        print("\n x5 低倍镜下的结果：")
        t1 = 0.8
        false_positive_rate_x5, true_positive_rate_x5, roc_auc_x5 = detector.evaluate(t1, cancer_map, mask_img)

        print("\n x20 高倍镜下增强的结果：")
        t2 = 0.8
        false_positive_rate_x20, true_positive_rate_x20, roc_auc_x20 = detector.evaluate(t2, cancer_map2, mask_img)

        print("\n x40 高倍镜下增强的结果：")
        t3 = 0.5
        false_positive_rate_x40, true_positive_rate_x40, roc_auc_x40 = detector.evaluate(t3, cancer_map3, mask_img)

        t4 = 0.5
        false_positive_rate_f, true_positive_rate_f, roc_auc_f = detector.evaluate(t4, cancer_map4, mask_img)

        fig, axes = plt.subplots(2, 4, figsize=(60, 40), dpi=100)
        ax = axes.ravel()

        ax[1].imshow(src_img)
        ax[1].set_title("src_img")

        ax[6].imshow(src_img)
        ax[6].imshow(mask_img, alpha=0.6)
        ax[6].set_title("mask_img")

        ax[0].set_title('Receiver Operating Characteristic')
        ax[0].plot(false_positive_rate_x5, true_positive_rate_x5, 'g',
                   label='x5  AUC = %0.4f' % roc_auc_x5)
        ax[0].plot(false_positive_rate_x20, true_positive_rate_x20, 'b',
                   label='x20 AUC = %0.4f' % roc_auc_x20)
        ax[0].plot(false_positive_rate_x40, true_positive_rate_x40, 'c',
                   label='x40 AUC = %0.4f' % roc_auc_x40)
        ax[0].plot(false_positive_rate_f, true_positive_rate_f, 'r',
                   label='final AUC = %0.4f' % roc_auc_f)

        ax[0].legend(loc='lower right')
        ax[0].plot([0, 1], [0, 1], 'r--')
        ax[0].set_xlim([-0.1, 1.2])
        ax[0].set_ylim([-0.1, 1.2])
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_xlabel('False Positive Rate')

        ax[2].imshow(src_img)
        ax[2].imshow(cancer_map, alpha=0.6)
        ax[2].contour(cancer_map >= t1)
        ax[2].set_title("cancer_map, t = %s" % t1)

        ax[3].imshow(src_img)
        ax[3].imshow(cancer_map2, alpha=0.6)
        ax[3].contour(cancer_map >= t2)
        ax[3].set_title("cancer_map2, t = %s" % t2)

        ax[7].imshow(src_img)
        ax[7].imshow(cancer_map3, alpha=0.6)
        ax[7].contour(cancer_map3 >= t3)
        ax[7].set_title("cancer_map3, t = %s" % t3)

        ax[5].imshow(src_img)
        ax[5].imshow(cancer_map4, alpha=0.6)
        ax[5].contour(cancer_map4 > t4)
        ax[5].set_title("final cancer_map, t = %s" % t4)

        for a in ax.ravel():
            a.axis('off')
        ax[0].axis("on")

        plt.show()


    def test_01(self):
        test_set = [("001", 2100, 3800, 2400, 4000),
                    ("003", 2400, 4700, 2600, 4850)]
        id = 1
        roi = test_set[id]
        slice_id = roi[0]
        x1 = roi[1]
        y1 = roi[2]
        x2 = roi[3]
        y2 = roi[4]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_%s.tif" % slice_id,
                                 'Train_Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)
        seeds = detector.get_random_seeds(10, x1, x2, y1, y2, None)
        print(seeds)
        seeds = detector.get_random_seeds(10, x1, x2, y1, y2, np.random.rand(150, 200))
        print(seeds)
        print(np.array(seeds) - np.array([x1, y1]))
        # a = np.array([x - x1, y - y1 ] for x,y in seeds)
        # print(a)

    def test_02(self):
        def func(x, y):
            return x * np.exp(-x ** 2 - y ** 2)

        grid_x, grid_y = np.mgrid[-2: 2: 0.02, -2: 2: 0.02]
        points = 4 * np.random.rand(1000, 2) - 2
        values = func(points[:, 0], points[:, 1])

        from scipy.interpolate import griddata
        # grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
        # grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

        plt.subplot(121)
        plt.imshow(func(grid_x, grid_y).T, extent=(-2, 2, -2, 2), origin='lower')
        plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
        plt.title('Original')
        plt.subplot(122)
        plt.imshow(grid_z1.T, extent=(-2, 2, -2, 2), origin='lower')
        plt.title('Linear')
        plt.gcf().set_size_inches(6, 6)
        plt.show()

    def test_03(self):
        from xml.dom import minidom
        doc = minidom.Document()
        rootNode = doc.createElement("ASAP_Annotations")
        doc.appendChild(rootNode)

        AnnotationsNode = doc.createElement("Annotations")
        rootNode.appendChild(AnnotationsNode)

        # one contour
        AnnotationNode = doc.createElement("Annotation")
        AnnotationNode.setAttribute("Name", "_0")
        AnnotationNode.setAttribute("Type", "Polygon")
        AnnotationNode.setAttribute("PartOfGroup", "_0")
        AnnotationNode.setAttribute("Color", "#F4FA00")
        AnnotationsNode.appendChild(AnnotationNode)

        CoordinatesNode = doc.createElement("Coordinates")
        AnnotationNode.appendChild(CoordinatesNode)


        CoordinateNode = doc.createElement("Coordinate")
        CoordinateNode.setAttribute("Order", "0")
        CoordinateNode.setAttribute("X", "123")
        CoordinateNode.setAttribute("Y", "456")
        CoordinatesNode.appendChild(CoordinateNode)

        f = open("test_output.xml", "w")
        doc.writexml(f, encoding="utf-8")
        f.close()

    def test_adaptive_detect_region(self):
        # # train set
        # test_set = {1: (1, 2100, 3800, 2400, 4000),
        #             2: (3, 2400, 4700, 2600, 4850),  # 小的局部150 x 200
        #             3: (3, 2000, 4300, 2800, 4900),  # 600 x 800
        #             4: (3, 721, 3244, 3044, 5851),  # 全切片范围
        #             5: (44, 410, 2895, 2813, 6019),  #
        #             6: (47, 391, 2402, 2891, 4280),  #
        #             }
        #
        # id = 3
        # roi = test_set[id]
        # id = roi[0]
        # x1 = roi[1]
        # y1 = roi[2]
        # x2 = roi[3]
        # y2 = roi[4]

        x1, y1, x2, y2 = 0, 0, 0, 0
        id = 2
        slice_id = "Tumor_{:0>3d}".format(id)

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                                 'Train_Tumor/%s.xml' % slice_id, slice_id)

        detector = AdaptiveDetector(c, imgCone)

        if x2 * y2 == 0:
            eff_zone = imgCone.get_effective_zone(1.25)
            x1, y1, x2, y2 = imgCone.get_mask_min_rect(eff_zone)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        history = detector.process(x1, y1, x2, y2, 1.25, extract_scale=40, patch_size=256,
                                   max_iter_nums=100, batch_size=100,
                                   limit_sampling_density=3, enhanced=True)
        detector.save_result_history(x1, y1, x2, y2, 1.25, history)


#################################################################################################################
    def test_adaptive_detect_region_train_slice(self):
        # train set
        # train_list = [9, 11, 16, 26, 39, 47, 58, 68, 72, 76]
        # train_list = [11, 16, 26, 39, 47, 58, 68, 72, 76]
        train_list = [98] # [1,2,3,4,5,6,7,8,10,12,13,14,15,17,18,19,20]range(27,39)
        result = {}
        # 如果输出癌变概率图，并进行评估
        enable_evaluate = False

        for id in train_list:
            x1, y1, x2, y2 = 0, 0, 0, 0

            slice_id = "Tumor_{:0>3d}".format(id)

            c = Params()
            c.load_config_file(JSON_PATH)
            imgCone = ImageCone(c, Open_Slide())

            # 读取数字全扫描切片图像
            tag = imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                                     'Train_Tumor/%s.xml' % slice_id, slice_id)

            detector = AdaptiveDetector(c, imgCone)

            if x2 * y2 == 0:
                x1, y1, x2, y2 = detector.get_detection_rectangle()
                print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            history = detector.process(x1, y1, x2, y2, 1.25, extract_scale=40, patch_size=256,
                                                   max_iter_nums=20, batch_size=100,
                                                   limit_sampling_density=2, enhanced = True)

            detector.save_result_history(x1, y1, x2, y2, 1.25, history)

            if enable_evaluate:
                cmb = CancerMapBuilder(c, extract_scale=40, patch_size=256)
                cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

                src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
                mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

                levels = [0.3, 0.5, 0.6, 0.8]
                false_positive_rate, true_positive_rate, roc_auc, dice = Evaluation.evaluate_slice_map(cancer_map,
                                                                                                       mask_img,
                                                                                                       levels)
                result[slice_id] = (roc_auc, dice)

                # 存盘输出部分
                # self.show_results(cancer_map, dice, false_positive_rate, history, levels, mask_img, roc_auc, slice_id,
                #                   src_img, true_positive_rate)
                save_path = "{}/results/cancer_pic".format(c.PROJECT_ROOT)
                Evaluation.save_result_picture(slice_id, src_img, mask_img, cancer_map, history, roc_auc, levels, save_path)
                # detector.save_result_xml(x1, y1, 1.25, cancer_map, levels)

        for slice, (auc, dices) in result.items():
            print("#################{}###################".format(slice))
            for t, value in dices:
                print("threshold = {:.3f}, dice coef = {:.6f}".format(t, value))
            print("ROC auc: {:.6f}".format(auc))
            print("#################{}###################".format(slice))

#################################################################################################################

#################################################################################################################
    def test_adaptive_detect_region_test(self):
        # test test
        test_set = {1: ("001", 100, 100, 2600, 2700),  # 检测 dice =0.7811
                    1.1: ("001", 800, 1600, 1600, 2300),  # dice = 0.76084
                    16: ("016", 0, 200, 3250, 2900),  # dice = 0.92056
                    21: ("021", 0, 0, 0, 0),  # dice = 0.93743
                    26: ("026", 0, 0, 0, 0),  # 检测，dice c3= 0.7601
                    61: ("061", 0, 0, 0, 0),  # 检测, c3 = 0.75468
                    4: ("004", 0, 0, 0, 0),  # 检测，c3 = 2.5917e-05， 检测区域太小
                    8: ("008", 0, 0, 0, 0),  # 检测，c3 = 0.003159
                    10: ("010", 0, 0, 0, 0),  # 检测,c3 = 3.7647e-05
                    11: ("011", 0, 0, 0, 0),  # 检测,c3 = 0.0005543
                    13: ("013", 0, 0, 0, 0),  # 检测,c3 = 0.003278
                    27: ("027", 0, 0, 0, 0),  # 检测,c3 = 0.9601, 0.3540
                    29: ("029", 0, 0, 0, 0),  # 检测,c3 =
                    30: ("030", 0, 0, 0, 0), # 检测,c3 = 0.02823
                    33: ("033", 0, 0, 0, 0), # 检测,c3 =
                    38: ("038", 0, 0, 0, 0), # 检测,c3 = 0.000214
                    40: ("040", 0, 0, 0, 0), # 检测,c3 = 0.06608, 检测错误
                    46: ("046", 0, 0, 0, 0), # 检测,c3 = 0.0005763
                    48: ("048", 0, 0, 0, 0), # 检测,c3 =
                    51: ("051", 0, 0, 0, 0), # 检测,c3 = 0.1478
                    52: ("052", 0, 0, 0, 0), # 检测,c3 =
                    71: ("071", 0, 0, 0, 0), # 检测,c3 =
                    64: ("064", 0, 0, 0, 0), # 检测,c3 =
                    65: ("065", 0, 0, 0, 0), # 检测,c3 =
                    66: ("066", 0, 0, 0, 0), # 检测,c3 =
                    68: ("068", 0, 0, 0, 0), # 检测,c3 =
                    69: ("069", 0, 0, 0, 0), # 检测,c3 =
                    73: ("073", 0, 0, 0, 0), # 检测,c3 =
                    }

        id = 39

        roi = test_set[id]
        slice_id = roi[0]
        x1 = roi[1]
        y1 = roi[2]
        x2 = roi[3]
        y2 = roi[4]

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        # tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
        #                          'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)
        tag = imgCone.open_slide("Testing/images/test_%s.tif" % slice_id,
                                 'Testing/images/test_%s.xml' % slice_id, "test_%s" % slice_id)

        detector = AdaptiveDetector(c, imgCone)

        if x2 * y2 == 0:
            eff_zone = imgCone.get_effective_zone(1.25)
            x1, y1, x2, y2 = imgCone.get_mask_min_rect(eff_zone)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        cancer_map, history = detector.process(x1, y1, x2, y2, 1.25, extract_scale = 40, patch_size = 256,
                                               max_iter_nums=100, batch_size=100,
                                               limit_sampling_density=1, )

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        levels = [0.2, 0.3, 0.5, 0.6, 0.8]
        false_positive_rate, true_positive_rate, roc_auc, dice = Evaluation.evaluate_slice_map(cancer_map, mask_img, levels)

        detector.save_result_history(x1, y1, x2, y2, 1.25, history)

        enable_show = False
    # 存盘输出部分
        if enable_show:
            save_path = "{}/results/cancer_pic".format(c.PROJECT_ROOT)
            Evaluation.save_result_picture(slice_id, src_img, mask_img, cancer_map, history, roc_auc, levels, save_path)

            detector.save_result_xml(x1, y1, 1.25, cancer_map, levels)



    def test_adaptive_detect_region_test_slice(self):
        # test set
        # test_list = [1,2,4,8,10,11,13,16,21,26,27,29,30,33,38,40,46,48,51,52,61,64,65,66,68,69,71,73,74,75,79,
        # 82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]
        # test_list = [4,10,29,30,33,38,48,66,79,84,99,102,116,117,122]
        test_list = [99,117]
        # test_list = [16,21,26,27,29,30,33,38,40,46,48,51,52,61,64,65,66,68,69,71,73,74,75,79,
        #              82,84,90,94,97,99,102,104,105,108,110,113,116,117,121,122]
        result = {}
        # 如果输出癌变概率图，并进行评估
        enable_evaluate = False

        for id in test_list:
            x1, y1, x2, y2 = 0, 0, 0, 0

            slice_id = "Test_{:0>3d}".format(id)

            c = Params()
            c.load_config_file(JSON_PATH)
            imgCone = ImageCone(c, Open_Slide())

            # 读取数字全扫描切片图像
            tag = imgCone.open_slide("Testing/images/%s.tif" % slice_id,
                                     'Testing/images/%s.xml' % slice_id, slice_id)

            detector = AdaptiveDetector(c, imgCone)

            if x2 * y2 == 0:
                x1, y1, x2, y2 = detector.get_detection_rectangle()
                print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            history = detector.process(x1, y1, x2, y2, 1.25, extract_scale=40, patch_size=256,
                                                   max_iter_nums=20, batch_size=100,
                                                   limit_sampling_density=2, enhanced = True)

            detector.save_result_history(x1, y1, x2, y2, 1.25, history)

            if enable_evaluate:
                cmb = CancerMapBuilder(c, extract_scale=40, patch_size=256)
                cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

                src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
                mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

                levels = [0.3, 0.5, 0.6, 0.8]
                false_positive_rate, true_positive_rate, roc_auc, dice = Evaluation.evaluate_slice_map(cancer_map,
                                                                                                       mask_img,
                                                                                                       levels)
                result[slice_id] = (roc_auc, dice)

                # 存盘输出部分
                # self.show_results(cancer_map, dice, false_positive_rate, history, levels, mask_img, roc_auc, slice_id,
                #                   src_img, true_positive_rate)
                save_path = "{}/results/cancer_pic".format(c.PROJECT_ROOT)
                Evaluation.save_result_picture(slice_id, src_img, mask_img, cancer_map, history, roc_auc, levels, save_path)
                # detector.save_result_xml(x1, y1, 1.25, cancer_map, levels)

        for slice, (auc, dices) in result.items():
            print("#################{}###################".format(slice))
            for t, value in dices:
                print("threshold = {:.3f}, dice coef = {:.6f}".format(t, value))
            print("ROC auc: {:.6f}".format(auc))
            print("#################{}###################".format(slice))