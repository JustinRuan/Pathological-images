#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-24'

"""
import time
from feature import FeatureExtractor
from skimage import io
from sklearn.svm import NuSVC
from sklearn.externals import joblib
from sklearn import metrics

class PatchFeature(object):
    def __init__(self, params):
        self._params = params
        return

    # # train_filename = "ZoneA_train.txt"
    # def loading_data(self, data_filename):
    #     '''
    #     从指定文件列表中，读入图像文件，并计算特征，和分类Tag
    #     :param data_filename: 图像文件的列表，前项是文件名，后项是tag
    #     :return: 特征向量集合，tag集合
    #     '''
    #     root_path = self._params.PATCHS_ROOT_PATH
    #     data_file = "{}/{}".format(root_path, data_filename)
    #
    #     fe = FeatureExtractor()
    #     features = []
    #     tags = []
    #     count = 0
    #
    #     f = open(data_file, "r")
    #     for line in f:
    #         items = line.split(" ")
    #         patch_file = "{}/{}".format(root_path, items[0])
    #         img = io.imread(patch_file, as_grey=False)
    #         tag = int(items[1])
    #         fvector = fe.extract_feature(img, "best")
    #
    #         features.append(fvector)
    #         tags.append(tag)
    #
    #         if (0 == count%200):
    #             print("{} extract feature >>> {}".format(time.asctime( time.localtime()), count))
    #         count += 1
    #
    #     f.close()
    #     return features, tags

    # x = features, y = tags
    def train_svm(self, X, y, patch_size, name):
        '''
        训练由精标区提取的图块所组成训练集的SVM，并存盘
        :param X: 特征向量
        :param y: 标签tag
        :return: SVM
        '''
        clf = NuSVC(nu=0.5, kernel='rbf', probability=False) #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        rf = clf.fit(X ,y)

        model_file = self._params.PROJECT_ROOT + "/models/svm_5x{}_{}.model".format(patch_size, name)
        joblib.dump(rf, model_file)
        return clf

    def load_svm_model(self, patch_size, name):
        '''
        从存盘文件中，加载SVM
        :return: 所加载的SVM
        '''
        model_file = self._params.PROJECT_ROOT + "/models/svm_5x{}_{}.model".format(patch_size, name)
        clf = joblib.load(model_file)

        return clf

    def test_svm(self, test_filename, patch_size, name):
        '''
        对SVM进行测试
        :param test_filename: 所使用的测试图像的列表文件
        :return: 所预测的概率性结果
        '''
        features, expected_tags = self.loading_data(test_filename)
        classifier = self.load_svm_model(patch_size, name)
        predicted_tags = classifier.predict(features)
        # predicted_result = classifier.predict_proba(features)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected_tags, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_tags, predicted_tags))
        return
        # return predicted_result