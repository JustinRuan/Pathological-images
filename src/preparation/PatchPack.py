#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import os
import shutil
import random
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
from preparation import PatchFeature
import numpy as np
from feature import FeatureExtractor
from skimage import io
from sklearn.model_selection import train_test_split
from core.util import read_csv_file

class PatchPack(object):
    def __init__(self, params):
        self._params = params

    # "stroma"     0
    # "cancer"     1
    # "lymph"      2
    # "edge"      3
    def loading_filename_tags(self, dir_code, tag):
        '''
        从包含指定字符串的目录中读取文件列表，并给定标记
        :param dir_code: 需要遍历的目录所包含的关键词
        :param tag: 该目录所对应的标记
        :return:
        '''
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

    def create_data_txt(self, data_tag, file_tag):
        '''
        生成样本文件的列表，存入txt中
        :param data_tag: 样本集
        :param file_tag: 生成的两个列表文件中所包含的代号
        :return: 生成.txt文件
        '''

        root_path = self._params.PATCHS_ROOT_PATH

        random.shuffle(data_tag)

        full_filename = "{0}/{1}.txt".format(root_path, file_tag)

        f = open(full_filename, "w")
        for item, tag in data_tag:
            f.write("{} {}\n".format(item, tag))
        f.close()

        return

    def initialize_sample_tags(self, dir_tag_map):
        '''
        从不同文件夹中加载不同标记的样本
        :param dir_tag_map: { "dir_code": tag }
        :return: 已经标注的，样本的文件路径
        '''
        data_tag = []

        for dir_code, tag in dir_tag_map.items():
            result = self.loading_filename_tags(dir_code, tag)
            data_tag.extend(result)

        return data_tag

    def refine_sample_tags_SVM(self, dir_map, train_code):
        '''
        精炼过程：迭代过程，从已经标注的样本中寻找最典经（最容易分类正确）的样本，并用它们训练SVM
        :param dir_map: 样本以及标记
        :param train_code: 生成SVM的代号
        :return: 得到一个能进行SC二分的分类器
        '''
        # train_code = "R_SC_5x128"
        data_tag = self.initialize_sample_tags(dir_map)
        self.create_data_txt(data_tag, train_code)

        # self.create_train_test_data(data_tag, 0.5, 0.5, train_code)
        #
        # pf = PatchFeature(self._params)
        # features_1, tags_1 = pf.loading_data("{}_train.txt".format(train_code))
        # features_2, tags_2 = pf.loading_data("{}_test.txt".format(train_code))
        pf = PatchFeature(self._params)
        features, tag = pf.loading_data("{}.txt".format(train_code))
        features_1, features_2, tags_1, tags_2 = train_test_split(features, tag, test_size=0.5, random_state=0)

        clf = SVC(C=1, kernel='rbf', probability=False) #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

        for i in range(3):
            rf = clf.fit(features_1, tags_1)
            predicted_tags = rf.predict(features_2)
            print("Classification report for classifier:\n%s\n"
                  % ( metrics.classification_report(tags_2, predicted_tags)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(tags_2, predicted_tags))

            simple_samples = (predicted_tags == tags_2)
            features_2 = np.array(features_2)[simple_samples,:]
            tags_2 = np.array(tags_2)[simple_samples]

            rf = clf.fit(features_2, tags_2)
            predicted_tags = rf.predict(features_1)
            print("Classification report for classifier %s:\n%s\n"
                  % (rf, metrics.classification_report(tags_1, predicted_tags)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(tags_1, predicted_tags))

            simple_samples = (predicted_tags == tags_1)
            features_1 = np.array(features_1)[simple_samples,:]
            tags_1 = np.array(tags_1)[simple_samples]

        model_file = self._params.PROJECT_ROOT + "/models/svm_{}.model".format(train_code)
        joblib.dump(rf, model_file)

        return

    def extract_refine_sample_SC(self, extract_scale, SC_dir_map, train_code, patch_size):
        '''
        用精炼SVM对样本进行分类，找出间质中的癌图块，以及癌变区域中的间质图块
        :param extract_scale: 提取的倍镜数
        :param SC_dir_map: 将要进行分类的S和C图块
        :param train_code: 生成新的文件的代码
        :param patch_size: 图块大小
        :return:
        '''
        model_file = self._params.PROJECT_ROOT + "/models/svm_{}.model".format(train_code)
        classifier = joblib.load(model_file)

        fe = FeatureExtractor()
        data_tag = self.initialize_sample_tags(SC_dir_map)

        ClearCancer = []
        ClearStroma = []
        AmbiguousCancer = []
        AmbiguousStroma = []

        Root_path = self._params.PATCHS_ROOT_PATH
        for patch_file, tag in data_tag:
            old_path = "{}/{}".format(Root_path, patch_file)
            img = io.imread(old_path)
            fvector = fe.extract_feature(img, "best")
            predicted_tags = classifier.predict([fvector])
            if (predicted_tags == tag):
                if tag == 0:
                    ClearStroma.append((patch_file, 0))
                elif tag == 1:
                    ClearCancer.append((patch_file, 1))
            else:
                if tag == 0:
                    AmbiguousStroma.append((patch_file, 1))
                elif tag == 1:
                    AmbiguousCancer.append((patch_file, 0))

        intScale = np.rint(extract_scale * 100).astype(np.int)
        pathCancer = "S{}_{}_{}".format(intScale,patch_size, "ClearCancer")
        pathStroma = "S{}_{}_{}".format(intScale,patch_size, "ClearStroma")
        pathAmbiguousCancer = "S{}_{}_{}".format(intScale, patch_size,"AmbiguousCancer")
        pathAmbiguousStroma = "S{}_{}_{}".format( intScale, patch_size,"AmbiguousStroma")

        self.create_data_txt(ClearCancer, pathCancer)
        self.create_data_txt(ClearStroma, pathStroma)
        self.create_data_txt(AmbiguousCancer, pathAmbiguousCancer)
        self.create_data_txt(AmbiguousStroma, pathAmbiguousStroma)

        return

    # def extract_refine_sample_LE(self, extract_scale, LE_dir_map, train_code, patch_size):
    #     '''
    #     用精炼SVM对样本进行分类，找出边缘区域中的癌图块和间质图块，以及淋巴区域中的间质图块和癌变图块
    #     :param extract_scale: 提取的倍镜数
    #     :param SC_dir_map: 将要进行分类的L和E图块
    #     :param train_code: 生成新的目录的代码
    #     :param patch_size: 图块大小
    #     :return:
    #     '''
    #     model_file = self._params.PROJECT_ROOT + "/models/svm_{}.model".format(train_code)
    #     classifier = joblib.load(model_file)
    #
    #     fe = FeatureExtractor()
    #     data_tag = self.initialize_sample_tags(LE_dir_map) # lymph 0, edge 1
    #
    #     Root_path = self._params.PATCHS_ROOT_PATH
    #     intScale = np.rint(extract_scale * 100).astype(np.int)
    #     pathLymphStroma = "{}/S{}_{}_{}".format(Root_path, intScale, patch_size,"LymphStroma")
    #     pathLymphCancer = "{}/S{}_{}_{}".format(Root_path,intScale, patch_size,"LymphCancer")
    #     pathEdgeStroma = "{}/S{}_{}_{}".format(Root_path, intScale, patch_size,"EdgeStroma")
    #     pathEdgeCancer = "{}/S{}_{}_{}".format(Root_path,intScale, patch_size,"EdgeCancer")
    #
    #     if (not os.path.exists(pathLymphStroma)):
    #         os.makedirs(pathLymphStroma)
    #
    #     if (not os.path.exists(pathLymphCancer)):
    #         os.makedirs(pathLymphCancer)
    #
    #     if (not os.path.exists(pathEdgeStroma)):
    #         os.makedirs(pathEdgeStroma)
    #
    #     if (not os.path.exists(pathEdgeCancer)):
    #         os.makedirs(pathEdgeCancer)
    #
    #     for patch_file, tag in data_tag:
    #         old_path = "{}/{}".format(Root_path, patch_file)
    #         img = io.imread(old_path, as_grey=False)
    #         fvector = fe.extract_feature(img, "best")
    #
    #         predicted_tags = classifier.predict([fvector])
    #         old_filename = os.path.split(patch_file)[-1]
    #         if (tag == 0): # Lymph
    #             if predicted_tags == 0:
    #                 new_filename = "{}/{}".format(pathLymphStroma, old_filename)
    #             elif predicted_tags == 1:
    #                 new_filename = "{}/{}".format(pathLymphCancer, old_filename)
    #         else:       # Edge
    #             if predicted_tags == 0:
    #                 new_filename = "{}/{}".format(pathEdgeStroma, old_filename)
    #             elif predicted_tags == 1:
    #                 new_filename = "{}/{}".format(pathEdgeCancer, old_filename)
    #         shutil.copy(old_path, new_filename)
    #     return

    def packing_refined_samples(self, extract_scale, patch_size):

        root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(extract_scale * 100).astype(np.int)
        pathCancer = "S{}_{}_{}.txt".format(intScale,patch_size, "ClearCancer")
        pathStroma = "S{}_{}_{}.txt".format(intScale,patch_size, "ClearStroma")
        pathAmbiguousCancer = "S{}_{}_{}.txt".format(intScale, patch_size,"AmbiguousCancer")
        pathAmbiguousStroma = "S{}_{}_{}.txt".format( intScale, patch_size,"AmbiguousStroma")

        #暧昧的癌变图块（癌变区中间质图块），暧昧的间质图块（间质区中的癌变图块）
        file_map = {pathCancer: 1, pathStroma: 0, pathAmbiguousCancer: 0, pathAmbiguousStroma: 1}

        data_tag = []

        for filename, tag in file_map.items():
            csv_path = "{}/{}".format(root_path, filename)
            f = open(csv_path, "r")
            lines = f.readlines()
            for line in lines:
                items = line.split(" ")
                tag = int(items[1])
                data_tag.append((items[0], tag))

        self.create_train_test_data(data_tag, 0.8, 0.2, "CNN_R_{}_{}".format(intScale, patch_size))

        return