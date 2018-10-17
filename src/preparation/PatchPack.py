#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-14'

"""

import os
import random
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
from preparation import PatchFeature
import numpy as np

class PatchPack(object):
    def __init__(self, params):
        self._params = params

    # "stroma"     0
    # "cancer"     1
    # "lymph"      2
    # "edge"      3
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

    def initialize_sample_tags_SCL(self, cancer_dirs, stroma_dirs, lymph_dirs):
        data_tag = []
        for s_dir in stroma_dirs:
            result = self.loading_filename_tags(s_dir, 0)
            data_tag.extend(result)

        for c_dir in cancer_dirs:
            result = self.loading_filename_tags(c_dir, 1)
            data_tag.extend(result)

        for l_dir in lymph_dirs:
            result = self.loading_filename_tags(l_dir, 2)
            data_tag.extend(result)

        return data_tag

    def refine_sample_tags_SCL(self, cancer_dirs, stroma_dir, lymph_dirs):

        data_tag = self.initialize_sample_tags_SCL(cancer_dirs, stroma_dir, lymph_dirs)
        self.create_train_test_data(data_tag, 0.5, 0.5, "R_SC_5x128")

        pf = PatchFeature.PatchFeature(self._params)
        features_1, tags_1 = pf.loading_data("R_SC_5x128_train.txt")
        features_2, tags_2 = pf.loading_data("R_SC_5x128_test.txt")

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


        model_file = self._params.PROJECT_ROOT + "/models/svm_R_5x128_SCL.model"
        joblib.dump(rf, model_file)

        return

    def extract_refine_sample_SCL(self, cancer_dirs, stroma_dir, lymph_dirs):

        model_file = self._params.PROJECT_ROOT + "/models/svm_R_5x128_SCL.model"
        rf = joblib.load(model_file)

        pf = PatchFeature.PatchFeature(self._params)
        data_tag = self.initialize_sample_tags_SCL(cancer_dirs, stroma_dir, lymph_dirs)



        return
