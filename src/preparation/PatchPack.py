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

    def create_train_test_data(self, pos_count, neg_count, t):
        root_path = self._params.PATCHS_ROOT_PATH

        random.shuffle(self.pos)
        random.shuffle(self.neg)

        train_data = self.pos[:pos_count]
        train_data.extend(self.neg[:neg_count])
        random.shuffle(train_data)

        test_data = self.pos[pos_count:]
        test_data.extend(self.neg[neg_count:])
        random.shuffle(test_data)

        full_filename = "{0}/{1}_{2}.txt".format(root_path, t,"train")

        f = open(full_filename, "w")
        for item, tag in train_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        full_filename = "{0}/{1}_{2}.txt".format(root_path, t,"test")

        f = open(full_filename, "w")
        for item, tag in test_data:
            f.write("{} {}\n".format(item, tag))
        f.close()

        return