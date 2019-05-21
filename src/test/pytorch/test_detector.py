#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""

import unittest
from core import *
import matplotlib.pyplot as plt
from pytorch.detector import Detector
import numpy as np
from skimage.segmentation import mark_boundaries

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

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

    def test_detect2(self):

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
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
                                 'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)

        seeds, predictions = detector.detect_region(x1, y1, x2, y2, 1.25, 5, 128, interval=64)
        new20_seeds, new20_predictions = detector.detect_region_detailed(seeds, predictions, 5, 128, 20, 256)
        new40_seeds, new40_predictions = detector.detect_region_detailed(new20_seeds, new20_predictions, 20, 256, 40,
                                                                         256)
        # print(predictions_deep)

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        np.savez("{}/data/cancer_predictions_{}".format(c.PROJECT_ROOT, id), src_img,
                 seeds, predictions, new20_seeds, new20_predictions, new40_seeds, new40_predictions)

        # fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
        # ax = axes.ravel()
        #
        # ax[0].imshow(src_img)
        # ax[0].set_title("src_img")
        #
        # ax[1].imshow(src_img)
        # ax[1].imshow(cancer_map, alpha=0.6)
        # ax[1].set_title("cancer_map")
        #
        # ax[2].imshow(src_img)
        # ax[2].imshow(cancer_map2, alpha=0.6)
        # ax[2].set_title("cancer_map2")
        #
        # for a in ax.ravel():
        #     a.axis('off')
        #
        # plt.show()

        return

    def test_show_result(self):
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

        D = np.load("{}/data/cancer_predictions_{}.npz".format(c.PROJECT_ROOT, slice_id))
        src_img = D['arr_0']
        seeds = D['arr_1']
        predictions = D['arr_2']
        new20_seeds = D['arr_3']
        new20_predictions = D['arr_4']
        new40_seeds = D['arr_5']
        new40_predictions = D['arr_6']

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
                                 'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

        detector = Detector(c, imgCone)
        print(detector.ImageHeight, detector.ImageWidth)
        detector.setting_detected_area(x1, y1, x2, y2, 1.25)
        cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 5, 1.25, seeds, predictions, 128,
                                                                     None, None)
        cancer_map2, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 20, 1.25, new20_seeds,
                                                                      new20_predictions,
                                                                      256, prob_map, count_map)
        cancer_map3, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, 40, 1.25, new40_seeds,
                                                                      new40_predictions,
                                                                      256, prob_map, count_map)

        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        label_map = np.load("label_map.npy")
        cancer_map4 = detector.create_cancer_map_superpixels(cancer_map3, label_map)

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
        tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
                                 'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)

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
        from visdom import Visdom
        viz = Visdom()

    def test_adaptive_detect_region(self):
        # # train set
        test_set = {1: ("001", 2100, 3800, 2400, 4000),
                    2: ("003", 2400, 4700, 2600, 4850),  # 小的局部150 x 200
                    3: ("003", 2000, 4300, 2800, 4900),  # 600 x 800
                    4: ("003", 721, 3244, 3044, 5851),  # 全切片范围
                    5: ("044", 410, 2895, 2813, 6019),  #
                    6: ("047", 391, 2402, 2891, 4280),  #
                    }
        # # test test
        # test_set = [("001", 100, 100, 2600, 2700),
        #             ("016", 0, 200, 3250, 2900),
        #             ("021", 0, 2400, 3000, 6500),
        #             ("001", 800, 1600, 1600, 2300),]

        id = 3
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
        # tag = imgCone.open_slide("Testing/images/test_%s.tif" % slice_id,
        #                          'Testing/images/test_%s.xml' % slice_id, "test_%s" % slice_id)

        detector = Detector(c, imgCone)

        # def adaptive_detect_region(self, x1, y1, x2, y2, coordinate_scale, extract_scale, patch_size,
        #                            iter_nums, batch_size, threshold):
        cancer_map, history = detector.adaptive_detect_region(x1, y1, x2, y2, 1.25, 40, 256, max_iter_nums=50,
                                                              batch_size=10, limit_sampling_density=10, use_post=True)
        # label_map = np.load("label_map.npy")
        # cancer_map2 = detector.create_cancer_map_superpixels(cancer_map, label_map)

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        t1 = 0.5
        false_positive_rate, true_positive_rate, roc_auc, dice = detector.evaluate(t1, cancer_map, mask_img)

        from visdom import Visdom
        viz = Visdom(env="main")
        pic_auc = viz.line(
            Y=true_positive_rate,
            X=false_positive_rate,
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'legend': ['AUC = %0.6f' % roc_auc, ],
                'xlabel': 'False Positive Rate',
                'ylabel': 'True Positive Rate',
                'title': 'Receiver Operating Characteristic',
            },
        )

        viz.line(
            Y=[0, 1], X=[0, 1],
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
                'dash': np.array(['dot']),  # 'solid', 'dash', 'dashdot'
            },
            name='y = x',
            win=pic_auc,
            update='insert',
        )

        fig, axes = plt.subplots(2, 2, figsize=(15, 20), dpi=100)
        ax = axes.ravel()

        ax[1].imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
        shape = src_img.shape
        ax[1].set_title("src_img {} x {}".format(shape[0], shape[1]))

        # ax[0].set_title('Receiver Operating Characteristic')
        # ax[0].plot(false_positive_rate, true_positive_rate, 'g',
        #          label='AUC = %0.6f' % roc_auc)
        #
        # ax[0].legend(loc='lower right')
        # ax[0].plot([0, 1], [0, 1], 'r--')
        # ax[0].set_xlim([-0.1, 1.2])
        # ax[0].set_ylim([-0.1, 1.2])
        # ax[0].set_ylabel('True Positive Rate')
        # ax[0].set_xlabel('False Positive Rate')

        ax[2].imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
        ax[2].imshow(cancer_map, alpha=0.3)
        ax[2].contour(cancer_map >= t1, alpha=0.6)
        ax[2].set_title("cancer_map, t = {}, dice = {:.6f}".format(t1, dice))

        point = np.array(list(history.keys()))
        ax[3].imshow(mask_img)
        ax[3].scatter(point[:, 0], point[:, 1], s=1, marker='o', alpha=0.9)
        total = shape[0] * shape[1]
        count = len(point)
        disp_text = "history, count = {:d}, ratio = {:.4e}".format(count, count / total)
        ax[3].set_title(disp_text)
        print(disp_text)

        for a in ax.ravel():
            a.axis('off')
        # ax[0].axis("on")

        plt.show()

    # def test_search_focus_area(self):
    #     test_set = [("001", 2100, 3800, 2400, 4000),
    #                 ("003", 2400, 4700, 2600, 4850),  # 小的局部150 x 200
    #                 ("003", 2000, 4300, 2800, 4900),  # 600 x 800
    #                 ("003", 721, 3244, 3044, 5851),  # 全切片范围
    #                 ("044", 410, 2895, 2813, 6019),
    #                 ]
    #     id = 2
    #     roi = test_set[id]
    #     slice_id = roi[0]
    #     x1 = roi[1]
    #     y1 = roi[2]
    #     x2 = roi[3]
    #     y2 = roi[4]
    #
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #     imgCone = ImageCone(c, Open_Slide())
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("Tumor/Tumor_%s.tif" % slice_id,
    #                              'Tumor/tumor_%s.xml' % slice_id, "Tumor_%s" % slice_id)
    #
    #     detector = Detector(c, imgCone)
    #     detector.setting_detected_area(x1, y1, x2, y2, 1.25)
    #
    #     saved_record = np.load("detect.npz")
    #     cancer_map = saved_record['arr_0']
    #     result = detector.search_focus_area(x1, y1, cancer_map, (0.9, 0.1))
    #     print(result)
    #     # src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
    #     # mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

    def test_adaptive_detect_region_test(self):
        # test test
        test_set = {1: ("001", 100, 100, 2600, 2700),  # 检测 dice =0.83
                    2: ("001", 800, 1600, 1600, 2300),  # dice = 0.91
                    3: ("016", 0, 200, 3250, 2900),  # 检测不出
                    4: ("021", 0, 2400, 3000, 6500),  # 检测 dice， c3 0.9, c9 0.95
                    5: ("026", 0, 0, 0, 0),  # 检测，dice c3= 0.11
                    6: ("061", 0, 0, 0, 0),  # 检测, c3 = 0.06,
                    7: ("004", 0, 0, 0, 0),  # 检测，c3 = 0
                    8: ("008", 0, 0, 0, 0),  # 检测，c3 = 0
                    9: ("010", 0, 0, 0, 0),  # 检测,c3 = 0
                    11: ("011", 0, 0, 0, 0),  # 检测,c3 = 0
                    12: ("013", 0, 0, 0, 0),  # 检测,c3 = 0
                    13: ("016", 0, 0, 0, 0),  # 检测,c3 =
                    14: ("027", 0, 0, 0, 0),  # 检测,c3 =
                    15: ("029", 0, 0, 0, 0),  # 检测,c3 =
                    16: ("030", 0, 0, 0, 0),
                    17: ("033", 0, 0, 0, 0),
                    18: ("038", 0, 0, 0, 0),
                    19: ("040", 0, 0, 0, 0),
                    20: ("046", 0, 0, 0, 0),
                    21: ("048", 0, 0, 0, 0),
                    22: ("051", 0, 0, 0, 0),
                    23: ("052", 0, 0, 0, 0),
                    24: ("071", 0, 0, 0, 0),
                    25: ("064", 0, 0, 0, 0),
                    26: ("065", 0, 0, 0, 0),
                    27: ("066", 0, 0, 0, 0),
                    28: ("068", 0, 0, 0, 0),
                    29: ("069", 0, 0, 0, 0),
                    30: ("073", 0, 0, 0, 0),
                    }

        id = 2
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

        detector = Detector(c, imgCone)

        if x2 * y2 == 0:
            print("y ", detector.ImageHeight, ", x ", detector.ImageWidth)
            y2 = detector.ImageHeight - 1
            x2 = detector.ImageWidth - 1
            y1 = 0
            x1 = 0

        cancer_map, history = detector.adaptive_detect_region(x1, y1, x2, y2, 1.25, 40, 256, max_iter_nums=50,
                                                              batch_size=20, limit_sampling_density=10, use_post=True)

        src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
        mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

        t1 = 0.5
        false_positive_rate, true_positive_rate, roc_auc, dice = detector.evaluate(t1, cancer_map, mask_img)

        from visdom import Visdom
        viz = Visdom(env="main")
        pic_auc = viz.line(
            Y=true_positive_rate,
            X=false_positive_rate,
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'legend': ['AUC = %0.6f' % roc_auc, ],
                'xlabel': 'False Positive Rate',
                'ylabel': 'True Positive Rate',
                'title': 'Receiver Operating Characteristic',
            },
        )

        viz.line(
            Y=[0, 1], X=[0, 1],
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
                'dash': np.array(['dot']),  # 'solid', 'dash', 'dashdot'
            },
            name='y = x',
            win=pic_auc,
            update='insert',
        )

        fig, axes = plt.subplots(2, 2, figsize=(15, 20), dpi=100)
        ax = axes.ravel()

        ax[1].imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
        shape = src_img.shape
        ax[1].set_title("src_img {} x {}".format(shape[0], shape[1]))

        ax[2].imshow(mark_boundaries(src_img, mask_img, color=(1, 0, 0), ))
        ax[2].imshow(cancer_map, alpha=0.3)
        ax[2].contour(cancer_map >= t1, alpha=0.6)
        ax[2].set_title("cancer_map, t = {}, dice = {:.6f}".format(t1, dice))

        point = np.array(list(history.keys()))
        ax[3].imshow(mask_img)
        ax[3].scatter(point[:, 0], point[:, 1], s=1, marker='o', alpha=0.9)
        total = shape[0] * shape[1]
        count = len(point)
        disp_text = "history, count = {:d}, ratio = {:.4e}".format(count, count / total)
        ax[3].set_title(disp_text)
        print(disp_text)

        for a in ax.ravel():
            a.axis('off')
        # ax[0].axis("on")

        plt.show()