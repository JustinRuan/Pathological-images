#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import os
import random

class PatchPack(object):
    def __init__(self, params):
        self._params = params

    # tagPos = "normal"     0
    # tagNeg = "cancer"     1
    # tagEdge = "edge"      2
    def loading_filename_tags(self, dir_code, tag):

        root_path = self._params.PATCHS_ROOT_PATH
        result = []

        for dir_name in os.listdir(root_path):
            full_path = "{}/{}".format(root_path, dir_name)
            if dir_code in dir_name:
                filename_tags = self.get_filename(full_path, tag)
                result.extend(filename_tags)

        return result

    def get_filename(self, full_dir, tag):
        '''
        生成样本列表文件所需要的文件名的格式，即去掉路径中的PATCHS_ROOT_PATH部分
        :param full_dir: 完整路径的文件名
        :param tag: 标签
        :return: （相对路径的文件名，Tag）的集合
        '''
        root_path = self._params.PATCHS_ROOT_PATH
        right_len = len(root_path) + 1
        L = []
        for root, dirs, files in os.walk(full_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    file_path = os.path.join(root, file)
                    rfile = file_path.replace('\\', '/')[right_len:]
                    L.append((rfile, tag))
        return L

    def create_train_test_data(self, data_tag, train_size, test_size, file_tag):
        if (train_size + test_size > 1):
            return

        root_path = self._params.PATCHS_ROOT_PATH

        count = len(data_tag)
        train_count = int(train_size * count)
        test_count = int(test_size * count)

        random.shuffle(data_tag)
        train_data = data_tag[:train_count]
        test_data = data_tag[train_count : train_count + test_count]

        full_filename = "{0}/{1}_{2}.txt".format(root_path, file_tag,"train")

        f = open(full_filename, "w")
        for item, tag in train_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        full_filename = "{0}/{1}_{2}.txt".format(root_path, file_tag,"test")

        f = open(full_filename, "w")
        for item, tag in test_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        return
