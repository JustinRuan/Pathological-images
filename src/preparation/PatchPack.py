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
from preparation.normalization import ImageNormalization

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

    def extract_feature_save_file(self, train_file_code):
        pf = FeatureExtractor(self._params)
        features, tag = pf.extract_features_by_file_list("{}.txt".format(train_file_code), features_name = "most")

        data_filename = "{}/data/{}_features_tags".format(self._params.PROJECT_ROOT, train_file_code)
        np.savez(data_filename, features, tag)
        return

    def train_SVM(self, train_file_code):
        data_filename = "{}/data/{}_features_tags.npz".format(self._params.PROJECT_ROOT, train_file_code)
        D = np.load(data_filename)
        features = D['arr_0']
        tag = D['arr_1']

        X_scaled = preprocessing.scale(features)
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # X_scaled = min_max_scaler.fit_transform(features)
        # x_new = SelectKBest(k=12).fit_transform(X_scaled, tag)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, tag, test_size=0.3, random_state=0)

        clf = SVC(C=1, kernel='rbf', probability=False) #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

        rf = clf.fit(x_train, y_train)
        predicted_tags = rf.predict(x_test)
        print("Classification report for classifier:\n%s\n"
              % (metrics.classification_report(y_test, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted_tags))

    def train_SVM_for_refine_sample(self, train_file_code):
        '''
        精炼过程：迭代过程，从已经标注的样本中寻找最典经（最容易分类正确）的样本，并用它们训练SVM
        :param dir_map: 样本以及标记
        :param train_code: 生成SVM的代号
        :return: 得到一个能进行SC二分的分类器
        '''

        data_filename = "{}/data/{}_features_tags.npz".format(self._params.PROJECT_ROOT, train_file_code)
        D = np.load(data_filename)
        features = D['arr_0']
        tag = D['arr_1']

        # X_scaled = preprocessing.scale(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        features_1, features_2, tags_1, tags_2 = train_test_split(X_scaled, tag, test_size=0.5, random_state=0)

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

        model_file = self._params.PROJECT_ROOT + "/models/svm_{}.model".format(train_file_code)
        joblib.dump(rf, model_file)

        predicted_tags = rf.predict(X_scaled)
        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(tag, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(tag, predicted_tags))
        return

    def create_refined_sample_txt(self, extract_scale, patch_size, dir_map, tag_name_map, train_file_code):
        '''
        用精炼SVM对样本进行分类，找出间质中的癌图块，以及癌变区域中的间质图块
        :param extract_scale: 提取的倍镜数
        :param SC_dir_map: 将要进行分类的S和C图块
        :param train_code: 生成新的文件的代码
        :param patch_size: 图块大小
        :return:
        '''

        data_filename = "{}/data/{}_features_tags.npz".format(self._params.PROJECT_ROOT, train_file_code)
        D = np.load(data_filename)
        features = D['arr_0']
        scaler = StandardScaler()
        scaler.fit(features)

        model_file = self._params.PROJECT_ROOT + "/models/svm_{}.model".format(train_file_code)
        classifier = joblib.load(model_file)

        fe = FeatureExtractor(self._params)
        data_tag = self.initialize_sample_tags(dir_map)

        count = 0
        result_filenames = {}
        for tag, name in tag_name_map.items():
            result_filenames["True_" + name] = []
            result_filenames["False_" + name] = []

        Root_path = self._params.PATCHS_ROOT_PATH
        for patch_file, tag in data_tag:
            old_path = "{}/{}".format(Root_path, patch_file)
            img = io.imread(old_path)
            normal_img = ImageNormalization.normalize_mean(img)
            fvector = fe.extract_feature(normal_img, "most")
            f_scaled = scaler.transform([fvector])
            predicted_tags = classifier.predict(f_scaled)
            if (predicted_tags == tag):
                result_filenames["True_" + tag_name_map[tag]].append((patch_file, tag))
            else:
                # result_filenames["False_" + tag_name_map[tag]].append((patch_file, predicted_tags[0]))
                result_filenames["False_" + tag_name_map[tag]].append((patch_file, tag))

            if (0 == count%200):
                print("{} predicting  >>> {}".format(time.asctime( time.localtime()), count))
            count += 1

        intScale = np.rint(extract_scale * 100).astype(np.int)
        for name, result_list in result_filenames.items():
            path = "S{}_{}_{}".format(intScale,patch_size, name)
            self.create_data_txt(result_list, path)

        return

    def packing_refined_samples(self, file_map, extract_scale, patch_size):

        root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(extract_scale * 100).astype(np.int)

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