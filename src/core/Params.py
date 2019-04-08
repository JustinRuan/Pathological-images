#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-13'

"""
import os
import json
from core.util import get_project_root

# # 注意设置 项目的工作目录为src的上一级目录
# CONFIG_PATH = os.getcwd() + "\\config\\"


class Params(object):
    def __init__(self):
        # self.EXTRACT_SCALE = 0
        self.GLOBAL_SCALE = 0
        # self.PATCH_SIZE_HIGH = 0
        # self.PATCH_SIZE_LOW = 0

    def load_config_file(self, filename):
        '''
        读取配置文件
        :param filename: 配置文件名
        :return:
        '''
        # 读取数据
        with open(filename, 'r') as f:
            data = json.load(f)
        self.GLOBAL_SCALE = data[0]['GLOBAL_SCALE']

        self.KFB_SDK_PATH = data[1]['KFB_SDK_PATH']
        self.SLICES_ROOT_PATH = data[1]['SLICES_ROOT_PATH']
        # self.PATCHS_ROOT_PATH = data[1]['PATCHS_ROOT_PATH']
        self.PROJECT_ROOT = get_project_root()

        self.PATCHS_ROOT_PATH = dict(data[2])

        self.NUM_WORKERS = data[3]['NUM_WORKERS']

        return

    def save_default_value(self, filename):
        '''
        生成一个默认的配置文件，以便进行修改
        :param filename: 存盘的文件名
        :return:
        '''
        filePath = get_project_root() + "\\config\\" + filename
        data = (
            {
                'GLOBAL_SCALE': 1.25, },
            {
                'KFB_SDK_PATH': 'D:/CloudSpace/WorkSpace/lib/KFB_SDK',
                'SLICES_ROOT_PATH': 'D:/Study/breast/3Plus',
            },
            {
                "P0404": "D:/Data/Patches/P0404",
                "P0327": "D:/Data/Patches/P0327"
            },
            {
                "NUM_WORKERS": 1,
            }
        )

        # 写入 JSON 数据
        with open(filePath, 'w') as f:
            json.dump(data, f)

        return
