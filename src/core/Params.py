#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-16'

"""
import os
import json

# 注意设置 项目的工作目录为src的上一级目录
CONFIG_PATH = os.getcwd() + "\\config\\"


class Params(object):
    def __init__(self):
        self.EXTRACT_SCALE = 0
        self.PATCH_SIZE_HIGH = 0
        self.PATCH_SIZE_LOW = 0

    def load_config_file(self, filename):
        # 读取数据
        with open(filename, 'r') as f:
            data = json.load(f)

        self.EXTRACT_SCALE = data[0]['EXTRACT_SCALE']
        self.PATCH_SIZE_HIGH = data[0]['PATCH_SIZE_HIGH']
        self.PATCH_SIZE_LOW = data[0]['PATCH_SIZE_LOW']
        self.GLOBAL_SCALE = self.PATCH_SIZE_LOW / self.PATCH_SIZE_HIGH * self.EXTRACT_SCALE  # when googleNet， = 1.25
        self.AMPLIFICATION_SCALE = self.PATCH_SIZE_HIGH / self.PATCH_SIZE_LOW  # when googleNet， = 16

        self.KFB_SDK_PATH = data[1]['KFB_SDK_PATH']
        self.SLICES_ROOT_PATH = data[1]['SLICES_ROOT_PATH']
        self.PATCHS_ROOT_PATH = data[1]['PATCHS_ROOT_PATH']

        return

    def save_default_value(self, filename):
        filePath = CONFIG_PATH + filename
        data = (
            {'EXTRACT_SCALE': 20,
             'PATCH_SIZE_HIGH': 256,
             'PATCH_SIZE_LOW': 16,
             'EXTRACT_PATCH_DIST': 4,
             'CLASSIFY_PATCH_DIST': 8},
            {'KFB_SDK_PATH': 'D:/CloudSpace/DoingNow/WorkSpace/lib/KFB_SDK',
             'SLICES_ROOT_PATH': 'D:/Study/breast/3Plus',
             'PATCHS_ROOT_PATH': 'D:/Study/breast/Patches/S20'
             }
        )

        # 写入 JSON 数据
        with open(filePath, 'w') as f:
            json.dump(data, f)

        return
