#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import os
import utils
import random


def file_name(file_dir, sub_dir):
    L = []
    right_len = len(file_dir) + 1
    full_dir = file_dir + '/' + sub_dir
    for root, dirs, files in os.walk(full_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                file_path = os.path.join(root, file)
                rfile = file_path.replace('\\', '/')[right_len:]
                if rfile.find("normal") > 0:
                    tag = 0
                else:
                    tag = 1
                L.append((rfile, tag))
    return L


def save_random(file_list, file_dir, sub_dir):
    random.shuffle(file_list)
    filename = "{0}/{1}/{1}.txt".format(file_dir, sub_dir)

    f = open(filename, "w")
    for item, tag in file_list:
        f.write("{} {}\n".format(item, tag))
    f.close()

    return


if __name__ == '__main__':
    file_list = file_name(utils.TRAIN_TEST_PATCHES_PATH, "test")
    save_random(file_list, utils.TRAIN_TEST_PATCHES_PATH, "test")

    file_list = file_name(utils.TRAIN_TEST_PATCHES_PATH, "train")
    save_random(file_list, utils.TRAIN_TEST_PATCHES_PATH, "train")

    print("done!")
