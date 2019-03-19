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

class TestPatchSampler(unittest.TestCase):

   #     # 提取Cancer切片的编号：1~50

    def test_patch_openslide_cancer(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        code = "001"
        tag = imgCone.open_slide("Tumor/Tumor_{}.tif".format(code),
                                 'Tumor/tumor_{}.xml'.format(code), "Tumor_{}".format(code))
        self.assertTrue(tag)

        if tag:
            patch_size = 256
            extract_scale = 10

            ps = PatchSampler(c)

            patch_spacing = 200
            while True:
                c_seeds, e_seeds = ps.detect_cancer_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing)
                print("slide code = ", code, ", cancer_seeds = ", len(c_seeds), ", edge_seeds = ", len(e_seeds))

                print("是否提取图块？Y/N")
                tag_c = input()
                if tag_c == "Y":
                    seeds_dict = ps.get_multi_scale_seeds([20, 40], c_seeds, extract_scale)
                    ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "cancer")

                    seeds_dict2 = ps.get_multi_scale_seeds([20, 40], e_seeds, extract_scale)
                    ps.extract_patches_multi_scale(imgCone, seeds_dict2, patch_size, "edge")
                    break
                else:
                    print("输入癌变区域图块的提取间隔：")
                    patch_spacing = int(input())

            print("%s 完成" % code)
            return

   # 提取Normal切片的编号：1~50
    def test_patch_openslide_normal(self):
        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        code = "001"
        tag = imgCone.open_slide("Normal/Normal_{}.tif".format(code),
                                None, "Normal_{}".format(code))
        self.assertTrue(tag)

        if tag:
            patch_size = 256
            extract_scale = 10

            ps = PatchSampler(c)

            patch_spacing = 400
            while True:
                n_seeds = ps.detect_normal_patches_with_scale(imgCone, extract_scale, patch_size, patch_spacing)
                print("slide code = ", code, ", normal_seeds = ", len(n_seeds))

                print("是否提取图块？Y/N")
                tag_c = input()
                if tag_c == "Y":
                    seeds_dict = ps.get_multi_scale_seeds([20, 40], n_seeds, extract_scale)
                    ps.extract_patches_multi_scale(imgCone, seeds_dict, patch_size, "normal")
                    break
                else:
                    print("输入癌变区域图块的提取间隔：")
                    patch_spacing = int(input())

            print("%s 完成" % code)
            return
