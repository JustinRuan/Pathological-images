#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import time
import os
import shutil
import random
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np
from feature import FeatureExtractor
from skimage import io
from sklearn.model_selection import train_test_split
from core.util import read_csv_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from preparation.normalization import HistNormalization
from core.Block import Block
from core.ImageCone import ImageCone
from core.open_slide import Open_Slide
import matplotlib.pyplot as plt

class PatchPack(object):
    def __init__(self, params):
        self._params = params
        self.patch_db = None

    def loading_filename_tags(self, dir_code, tag):
        '''
        从包含指定字符串的目录中读取文件列表，并给定标记
        :param dir_code: 需要遍历的目录所包含的关键词
        :param tag: 该目录所对应的标记
        :return:
        '''
        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]
        result = []

        for dir_name in os.listdir(root_path):
            full_path = "{}/{}".format(root_path, dir_name)
            if dir_code == dir_name:
                filename_tags = self.get_filename(full_path, tag)
                # filename_tags = self.get_filename_mask_ratio(full_path, tag)
                result.extend(filename_tags)

        return result

    def get_filename(self, full_dir, tag):
        '''
        生成样本列表文件所需要的文件名的格式，即去掉路径中的PATCHS_ROOT_PATH部分
        :param full_dir: 完整路径的文件名
        :param tag: 标签
        :return: （相对路径的文件名，Tag）的集合
        '''
        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]
        right_len = len(root_path) + 1
        L = []
        for root, dirs, files in os.walk(full_dir):
            for file in files:
                if os.path.splitext(file)[1] in ['.jpg', '.png'] :
                    file_path = os.path.join(root, file)
                    rfile = file_path.replace('\\', '/')[right_len:]
                    L.append((rfile, tag))
        return L

    def create_train_test_data(self, data_tag, train_size, test_size, file_tag, need_balance):
        '''
        生成样本文件的列表，存入txt中
        :param data_tag: 样本集
        :param train_size: 训练集所占比例
        :param test_size: 测试集所占比例
        :param file_tag: 生成的两个列表文件中所包含的代号
        :return: 生成train.txt和test.txt
        '''
        if (train_size + test_size > 1):
            return

        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]

        # 平衡各个类别样本的数量, normal 远多于Cancer
        normal_data = data_tag[0]
        # normal_count = len(normal_data)
        cancer_data = data_tag[1]

        cancer_count = len(cancer_data)
        if need_balance:
            random.shuffle(normal_data)
            normal_data = normal_data[:cancer_count]

        normal_count = len(normal_data)

        print("count of Cancer:", cancer_count, "count of Normal:", normal_count, "ratio:", float(cancer_count)/normal_count)
        prepared_data = cancer_data
        prepared_data.extend(normal_data)
        count = len(prepared_data)
        train_count = int(train_size * count)
        test_count = int(test_size * count)

        random.shuffle(prepared_data)

        train_data = prepared_data[:train_count]
        test_data = prepared_data[train_count : train_count + test_count]

        for file_type, data in zip(["train", "test"], [train_data, test_data]):
            if len(data) == 0:
                continue

            full_filename = "{0}/{1}_{2}.txt".format(root_path, file_tag, file_type)

            f = open(full_filename, "w")
            for item, tag in data:
                if isinstance(tag, tuple):
                    str_tag = ""
                    for sub_tag in tag:
                        str_tag += " {}".format(sub_tag)

                    f.write("{}{}\n".format(item, str_tag))
                else:
                    f.write("{} {}\n".format(item, tag))
            f.close()

        # full_filename = "{0}/{1}_{2}.txt".format(root_path, file_tag,"test")
        #
        # f = open(full_filename, "w")
        # for item, tag in test_data:
        #     f.write("{} {}\n".format(item, tag))
        # f.close()

        return

    # def create_data_txt(self, data_tag, file_tag):
    #     '''
    #     生成样本文件的列表，存入txt中
    #     :param data_tag: 样本集
    #     :param file_tag: 生成的两个列表文件中所包含的代号
    #     :return: 生成.txt文件
    #     '''
    #
    #     root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]
    #
    #     # random.shuffle(data_tag)
    #
    #     full_filename = "{0}/{1}.txt".format(root_path, file_tag)
    #
    #     f = open(full_filename, "w")
    #     for item, tag in data_tag:
    #         f.write("{} {}\n".format(item, tag))
    #     f.close()
    #
    #     return

    def initialize_sample_tags(self, patches_code, dir_tag_map):
        '''
        从不同文件夹中加载不同标记的样本
        :param dir_tag_map: { "dir_code": tag }
        :return: 已经标注的，样本的文件路径
        '''
        data_tag = {0:[], 1:[]}
        self.patches_code = patches_code
        for dir_code, tag in dir_tag_map.items():
            result = self.loading_filename_tags(dir_code, tag)
            data_tag[tag].extend(result)

        return data_tag

    def filtering(self, data_tag, filter_mask):
        normal_data = data_tag[0]
        cancer_data = data_tag[1]

        filtered_cancer = []
        for rfile, tag in cancer_data:
            frags = rfile.split('/')
            if frags[1] not in filter_mask:
                filtered_cancer.append((rfile, tag))
        data_tag[1] = filtered_cancer

        filtered_normal = []
        for rfile, tag in normal_data:
            frags = rfile.split('/')
            if frags[1] not in filter_mask or "Normal" in frags[1]:
                filtered_normal.append((rfile, tag))
        data_tag[0] = filtered_normal

        return data_tag

    # ###############################################################################################################
    # # Multiple scale combination (MSC)
    # ###############################################################################################################
    # def create_train_test_data_MSC(self, scale_tag, data_tag, train_size, test_size, file_tag):
    #     if (train_size + test_size > 1):
    #         return
    #
    #     dir_x10 = scale_tag[10]
    #     dir_x20 = scale_tag[20]
    #     dir_x40 = scale_tag[40]
    #
    #     dir_x10_tag = {}
    #     for key, value in data_tag.items():
    #         full_dir_x10 = "{}_{}".format(dir_x10, key)
    #         dir_x10_tag[full_dir_x10] = value
    #
    #     data_x10_tag = self.initialize_sample_tags(dir_x10_tag)
    #     random.shuffle(data_x10_tag)
    #
    #     data_x20_tag = self.get_msc_filenames(dir_x10, dir_x20, data_x10_tag, 2)
    #     data_x40_tag = self.get_msc_filenames(dir_x10, dir_x40, data_x10_tag, 4)
    #
    #     new_file_tag = "{}_{}".format(file_tag, dir_x10)
    #     self.create_train_test_data(data_x10_tag, train_size, test_size, new_file_tag, suffle=False)
    #
    #     new_file_tag = "{}_{}".format(file_tag, dir_x20)
    #     self.create_train_test_data(data_x20_tag, train_size, test_size, new_file_tag, suffle=False)
    #
    #     new_file_tag = "{}_{}".format(file_tag, dir_x40)
    #     self.create_train_test_data(data_x40_tag, train_size, test_size, new_file_tag, suffle=False)
    #
    #     return
    #
    # def get_msc_filenames(self, scale_x10_tag, scale_other_tag, data_x10_tag, multiple):
    #     data_tag = []
    #     temp_block = Block()
    #     for pathname, tag in data_x10_tag:
    #         [dirname, filename]=os.path.split(pathname)
    #         dirname = dirname.replace(scale_x10_tag, scale_other_tag)
    #         temp_block.decoding(filename, 256, 256)
    #         temp_block.x = int(multiple * temp_block.x)
    #         temp_block.y = int(multiple * temp_block.y)
    #         temp_block.scale = multiple * temp_block.scale
    #         new_filename = "{}/{}.jpg".format(dirname, temp_block.encoding())
    #         data_tag.append((new_filename, tag))
    #
    #     return data_tag


    def calc_patch_cancer_ratio(self, patches_code, directory_list):
        self.patches_code = patches_code
        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]

        slice_db = {}
        for directory in directory_list:
            path = "{}/{}".format(root_path, directory)
            for (dirpath, dirnames, filenames) in os.walk(path):
                if len(filenames) == 0:
                    for subdir_name in dirnames:
                        if subdir_name not in slice_db:
                            slice_db[subdir_name] = []
                else:
                    slice_id = os.path.basename(dirpath)
                    slice_db[slice_id].extend(filenames)

        imgCone = ImageCone(self._params, Open_Slide())
        b = Block()
        patch_db = {}
        for slice_id, filenames in slice_db.items():
            imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                               'Train_Tumor/%s.xml' % slice_id, slice_id)
            max_w, max_h = imgCone.get_image_width_height_byScale(self._params.GLOBAL_SCALE)
            all_masks = imgCone.create_mask_image(self._params.GLOBAL_SCALE, 0)
            mask = all_masks['C']
            print("{}, area of mask = {}".format(slice_id, np.sum(mask)))

            for file_name in filenames:
                b.decoding(file_name, 256, 256)
                assert b.slice_number == slice_id, "Error: slice_number <> slice_id"
                # 提取在1.25倍镜下的中心坐标和边长
                xx = np.rint(b.x * self._params.GLOBAL_SCALE / b.scale).astype(np.int)
                yy = np.rint(b.y * self._params.GLOBAL_SCALE / b.scale).astype(np.int)
                patch_size = np.rint(b.w * self._params.GLOBAL_SCALE / b.scale).astype(np.int)
                half_w = patch_size >> 1

                nx1 = max(xx - half_w, 0)
                nx2 = min(xx + half_w, max_w)
                ny1 = max(yy - half_w, 0)
                ny2 = min(yy + half_w, max_h)
                sub_mask = mask[ny1:ny2, nx1:nx2]

                r = np.sum(sub_mask).astype(np.float) / (patch_size * patch_size)
                if r > 0:
                    patch_db[file_name] = r

        np.save('{}/patch_mask_ratio.npy'.format(root_path), patch_db)
        return

    # def get_filename_mask_ratio(self, full_dir,):
    #     root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]
    #
    #     if self.patch_db is None:
    #         data = np.load('{}/patch_mask_ratio.npy'.format(root_path), allow_pickle=True)
    #         self.patch_db = data[()]
    #
    #     right_len = len(root_path) + 1
    #     L = []
    #     for root, dirs, files in os.walk(full_dir):
    #         for file in files:
    #             if os.path.splitext(file)[1] in ['.jpg', '.png'] :
    #                 file_path = os.path.join(root, file)
    #                 rfile = file_path.replace('\\', '/')[right_len:]
    #                 left = rfile.rfind("/")
    #                 filename = rfile[left + 1:]
    #                 mask_ratio = self.patch_db.get(filename, 0)
    #                 new_tag = self._create_multidim_label(mask_ratio)
    #                 L.append((rfile, new_tag))
    #     return L

    def _create_multidim_label(self, mask_ratio):
        # if mask_ratio is None:
        #     return 0
        if mask_ratio > 0.5:
            return 1
        else:
            return 0

    def initialize_sample_tags_byMask(self, patches_code, directory_list):
        '''
        从不同文件夹中加载不同标记的样本
        :param dir_tag_map: { "dir_code": tag }
        :return: 已经标注的，样本的文件路径
        '''
        data_tag = {0:[], 1:[]}
        self.patches_code = patches_code
        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]

        if self.patch_db is None:
            data = np.load('{}/patch_mask_ratio.npy'.format(root_path), allow_pickle=True)
            self.patch_db = data[()]

        right_len = len(root_path) + 1
        for directory in directory_list:
            full_dir = "{}/{}".format(root_path, directory)
            for root, dirs, files in os.walk(full_dir):
                for file in files:
                    if os.path.splitext(file)[1] in ['.jpg', '.png'] :
                        file_path = os.path.join(root, file)
                        rfile = file_path.replace('\\', '/')[right_len:]
                        left = rfile.rfind("/")
                        filename = rfile[left + 1:]
                        mask_ratio = self.patch_db.get(filename, 0)
                        new_tag = self._create_multidim_label(mask_ratio)
                        if new_tag == 1:
                            data_tag[1].append((rfile, new_tag))
                        else:
                            data_tag[0].append((rfile, new_tag))
        return data_tag

    ###############################################################################################################
    # Double scale combination (DSC)
    # 当x20下mask占比>= 50%，x40也 >= 50%，label = 1
    # 当x20下mask占比 < 50%，x40 >= 50%，label = 1
    # 当x20下mask占比 >= 50%，x40 < 50%，label = 1
    # 当x20下mask占比 < 50%，x40也 < 50%，label = 0
    # 执行 或 逻辑， label为三维，（x20，x40，xD）
    '''
    {40 : ["S4000_256_cancer", "S4000_256_normal", "S4000_256_normal2", "S4000_256_edgeinner", "S4000_256_edgeouter",],
     20 : [S2000_256_cancer", "S2000_256_normal", "S2000_256_normal2", "S2000_256_edgeinner", "S2000_256_edgeouter" ]
    '''
    ###############################################################################################################
    def create_train_test_data_DSC(self, patches_code, directory_list, sample_filename, source_scale, target_scale):

        new_data_tag = self.initialize_sample_tags_byMask(patches_code, directory_list)

        target_dict = {}
        for index in range(0,2):
            for item in new_data_tag[index]:
                [dirname, filename] = os.path.split(item[0])
                target_dict[filename] = (item[0], item[1])

        soruce_train_filename = "{}_train.txt".format(sample_filename)
        soruce_test_filename = "{}_test.txt".format(sample_filename)

        source_scale_code = "{:0>4}".format(source_scale * 100)
        target_scale_code = "{:0>4}_{:0>4}".format(source_scale * 100, target_scale * 100)

        target_train_filename = soruce_train_filename.replace(source_scale_code, target_scale_code)
        target_test_filename = soruce_test_filename.replace(source_scale_code, target_scale_code)

        multiple = target_scale / source_scale
        temp_block = Block()
        root_path = self._params.PATCHS_ROOT_PATH[self.patches_code]
        for source_filename, target_filename in [(soruce_train_filename, target_train_filename),
                                                 (soruce_test_filename, target_test_filename)]:

            source_file = open("{}/{}".format(root_path, source_filename), "r")
            target_file = open("{}/{}".format(root_path, target_filename), "w")
            lines = source_file.readlines()
            for line in lines:
                items = line.split(" ")
                [dirname, filename] = os.path.split(items[0])
                tag = int(items[1]) # 只能是单标签

                temp_block.decoding(filename, 256, 256)
                temp_block.x = int(multiple * temp_block.x)
                temp_block.y = int(multiple * temp_block.y)
                temp_block.scale = multiple * temp_block.scale
                new_filename = "{}.jpg".format(temp_block.encoding())

                another = target_dict[new_filename]
                H_tag = np.logical_or(tag, another[1]).astype(np.int)
                # 20倍镜下样本在前，40倍镜下样本在后，最后是混合
                if source_scale == 40:
                    target_file.write("{} {} {} {} {}\n".format(another[0], items[0], another[1], tag, H_tag))
                else:
                    target_file.write("{} {} {} {} {}\n".format(items[0], another[0], tag, another[1], H_tag))

            source_file.close()
            target_file.close()

        return

    # def get_msc_filenames(self, scale_x10_tag, scale_other_tag, data_x10_tag, multiple):
    #     data_tag = []
    #     temp_block = Block()
    #     for pathname, tag in data_x10_tag:
    #         [dirname, filename]=os.path.split(pathname)
    #         dirname = dirname.replace(scale_x10_tag, scale_other_tag)
    #         temp_block.decoding(filename, 256, 256)
    #         temp_block.x = int(multiple * temp_block.x)
    #         temp_block.y = int(multiple * temp_block.y)
    #         temp_block.scale = multiple * temp_block.scale
    #         new_filename = "{}/{}.jpg".format(dirname, temp_block.encoding())
    #         data_tag.append((new_filename, tag))
    #
    #     return data_tag