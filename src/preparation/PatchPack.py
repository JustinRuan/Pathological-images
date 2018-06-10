#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import os
import random

class PatchPack(object):
    def __init__(self, params):
        self._params = params

    # tagPos = "normalA"    0
    # tagNeg = "cancerA"    1
    def loading(self, tagPos, tagNeg):
        '''
        根据 设定的正例反例所在目录的关键字，生成对应的样本集合
        :param tagPos:  正例所在文件夹所包含的关键字
        :param tagNeg:  反例所在文件夹所包含的关键字
        :return: 正反例的数量
        '''
        root_path = self._params.PATCHS_ROOT_PATH

        self.pos = []
        self.neg = []

        for dir_name in os.listdir(root_path):
            full_path = "{}/{}".format(root_path, dir_name)
            if tagPos in dir_name:
                filename_tags = self.file_name(full_path, 0)
                self.pos.extend(filename_tags)
            elif tagNeg in dir_name:
                filename_tags = self.file_name(full_path, 1)
                self.neg.extend(filename_tags)

        return (len(self.pos), len(self.neg))

    def file_name(self, full_dir, tag):
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

    def create_train_test_data(self, pos_count, neg_count, pos_test_count, neg_test_count,
                               pos_check_count, neg_check_count, fileTag):
        '''
        根据已经读到的正例反例信息，按指定长度生成Train,Test, Check三个完全独立的样本集
        :param pos_count: train用正例数量
        :param neg_count: train用正例数量
        :param pos_test_count: test用正例数量
        :param neg_test_count: test用正例数量
        :param pos_check_count: check用正例数量
        :param neg_check_count: check用正例数量
        :param fileTag: 生成的文件列表，所包含的标识符
        :return: 生成三个存盘文件
        '''
        root_path = self._params.PATCHS_ROOT_PATH

        random.shuffle(self.pos)
        random.shuffle(self.neg)

        train_data = self.pos[:pos_count]
        train_data.extend(self.neg[:neg_count])
        random.shuffle(train_data)

        test_data = self.pos[pos_count : pos_count + pos_test_count]
        test_data.extend(self.neg[neg_count : neg_count + neg_test_count])
        random.shuffle(test_data)

        check_data = self.pos[pos_count + pos_test_count:]
        check_data.extend(self.neg[neg_count + neg_test_count:])
        random.shuffle(check_data)

        full_filename = "{0}/{1}_{2}.txt".format(root_path, fileTag,"train")

        f = open(full_filename, "w")
        for item, tag in train_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        full_filename = "{0}/{1}_{2}.txt".format(root_path, fileTag,"test")

        f = open(full_filename, "w")
        for item, tag in test_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        full_filename = "{0}/{1}_{2}.txt".format(root_path, fileTag,"check")

        f = open(full_filename, "w")
        for item, tag in check_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        return