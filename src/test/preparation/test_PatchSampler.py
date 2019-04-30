#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import unittest
from core import *
from preparation import *
import matplotlib.pyplot as plt

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestPatchSampler(unittest.TestCase):

   #     # 提取Cancer切片的编号：1~50

    def test_patch_openslide_cancer(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())
        patch_size = 256
        extract_scale = 40

        ps = PatchSampler(c)

        patch_spacing = 400

        for i in range(1, 3):
            code = "{:0>3d}".format(i)
            print("processing ", code, " ... ...")

            # 读取数字全扫描切片图像
            # code = "003"
            tag = imgCone.open_slide("Train_Tumor/Tumor_{}.tif".format(code),
                                     'Train_Tumor/tumor_{}.xml'.format(code), "Tumor_{}".format(code))
            self.assertTrue(tag)

            if tag:

                c_seeds, ei_seeds, eo_seeds, n_seeds = ps.detect_cancer_patches_with_scale(imgCone, extract_scale, patch_size,
                                                                                  patch_spacing, edge_width=8)
                print("slide code = ", code, ", cancer_seeds = ", len(c_seeds), ", normal_seeds = ", len(n_seeds),
                      ", inner edge_seeds = ", len(ei_seeds), ", outer edge_seeds = ", len(eo_seeds))

                # seeds_dict = ps.get_multi_scale_seeds([10, 20], c_seeds, extract_scale)
                seeds_dict = ps.get_multi_scale_seeds([], c_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "cancer", "P0430")

                seeds_dict4 = ps.get_multi_scale_seeds([], n_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict4, patch_size, "noraml", "P0430")

                # seeds_dict2 = ps.get_multi_scale_seeds([10, 20], ei_seeds, extract_scale)
                # ps.extract_patches_multi_scale(imgCone, seeds_dict2, patch_size, "edgeinner")
                #
                # seeds_dict3 = ps.get_multi_scale_seeds([10, 20], eo_seeds, extract_scale)
                # ps.extract_patches_multi_scale(imgCone, seeds_dict3, patch_size, "edgeouter")

                # while True:
                #     c_seeds, ei_seeds, eo_seeds = ps.detect_cancer_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing)
                #     print("slide code = ", code, ", cancer_seeds = ", len(c_seeds),
                #           ", inner edge_seeds = ", len(ei_seeds), ", outer edge_seeds = ", len(eo_seeds))
                #
                #     print("是否提取图块？Y/N")
                #     tag_c = input()
                #     if tag_c == "Y" or len(tag_c) == 0:
                #         seeds_dict = ps.get_multi_scale_seeds([10, 20], c_seeds, extract_scale)
                #         ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "cancer")
                #
                #         seeds_dict2 = ps.get_multi_scale_seeds([10, 20], ei_seeds, extract_scale)
                #         ps.extract_patches_multi_scale(imgCone, seeds_dict2, patch_size, "edgeinner")
                #
                #         seeds_dict3 = ps.get_multi_scale_seeds([10, 20], eo_seeds, extract_scale)
                #         ps.extract_patches_multi_scale(imgCone, seeds_dict3, patch_size, "edgeouter")
                #         break
                #     else:
                #         print("输入癌变区域图块的提取间隔：")
                #         patch_spacing = int(input())

                print("%s 完成" % code)
        return

   # 提取Normal切片的编号：1~50
    def test_patch_openslide_normal(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        patch_size = 256
        extract_scale = 40

        ps = PatchSampler(c)

        patch_spacing = 1200

        for i in range(1, 51):
            code = "{:0>3d}".format(i)
            print("processing ", code, " ... ...")

            # 读取数字全扫描切片图像
            # code = "001"
            tag = imgCone.open_slide("Train_Normal/Normal_{}.tif".format(code),
                                    None, "Normal_{}".format(code))
            self.assertTrue(tag)

            if tag:
                n_seeds = ps.detect_normal_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing)
                print("slide code = ", code, ", normal_seeds = ", len(n_seeds))

                # seeds_dict = ps.get_multi_scale_seeds([10, 20], n_seeds, extract_scale)
                seeds_dict = ps.get_multi_scale_seeds([], n_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "normal")

                # while True:
                #     n_seeds = ps.detect_normal_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing)
                #     print("slide code = ", code, ", normal_seeds = ", len(n_seeds))
                #
                #     print("是否提取图块？Y/N")
                #     tag_c = input()
                #     if tag_c == "Y" or len(tag_c) == 0:
                #         seeds_dict = ps.get_multi_scale_seeds([10, 20], n_seeds, extract_scale)
                #         ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "normal")
                #         break
                #     else:
                #         print("输入癌变区域图块的提取间隔：")
                #         patch_spacing = int(input())

                print("%s 完成" % code)
        return


   # 提取测试集切片的图块，用于预测评估,
   # 此图块不能用于分类器的训练，即不用使用 标注信息。
   # 但可以用来收集图块的均值，方差等统计特征，用于预处理
    def test_patch_openslide_cancer_normal_Eval(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())
        patch_size = 256
        extract_scale = 40

        ps = PatchSampler(c)

        patch_spacing_cancer = 400
        patch_spacing_normal = 2000

        slice_IDs = [16, 26, 61, 4, 8, 10, 11, 13]  # [1, 16, 26, 61, 4, 8, 10, 11, 13]
        # slice_IDs = [1]
        for i in slice_IDs:
            code = "{:0>3d}".format(i)
            print("processing ", code, " ... ...")

            # 读取数字全扫描切片图像
            tag = imgCone.open_slide("Testing/images/test_%s.tif" % code,
                                     'Testing/images/test_%s.xml' % code, "test_%s" % code)
            self.assertTrue(tag)

            if tag:

                c_seeds, ei_seeds, eo_seeds = ps.detect_cancer_patches_with_scale(imgCone, extract_scale, patch_size,
                                                                                  patch_spacing_cancer, edge_width=8)
                print("slide code = ", code, ", cancer_seeds = ", len(c_seeds),
                      ", inner edge_seeds = ", len(ei_seeds), ", outer edge_seeds = ", len(eo_seeds))

                seeds_dict = ps.get_multi_scale_seeds([10, 20], c_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "cancer", "P0330")

                seeds_dict2 = ps.get_multi_scale_seeds([10, 20], ei_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict2, patch_size, "edgeinner", "P0330")

                seeds_dict3 = ps.get_multi_scale_seeds([10, 20], eo_seeds, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict3, patch_size, "edgeouter", "P0330")

                mask = imgCone.create_mask_image(c.GLOBAL_SCALE, 8)

                C_mask = mask["C"]
                n_seeds = ps.detect_normal_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing_normal,
                                                              c_mask=C_mask)
                c_seeds = set(c_seeds)
                n_seeds = set(n_seeds)
                c_and_n = c_seeds & n_seeds

                result_n = list(n_seeds - c_and_n)

                print("slide code = ", code, ", normal_seeds = ", len(result_n))

                seeds_dict = ps.get_multi_scale_seeds([10, 20], result_n, extract_scale)
                ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "normal", "P0330")

                print("%s 完成" % code)
        return