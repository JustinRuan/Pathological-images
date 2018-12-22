#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-14'

"""

import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.utils.data as Data
from torch.autograd import Variable
from core.util import read_csv_file
from pytorch.image_dataset import Image_Dataset

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn import metrics

class Feature_Extractor(object):
    def __init__(self, params, model_name, patch_type):
        '''
        初始化
        :param params: 系统参数
        :param model_name: 提取特征所用的网络模型
        :param patch_type: 所处理的图块的类型
        '''
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        self.use_GPU = True

    def load_pretrained_model(self):
        '''
        加载imagenet训练好的模型
        :return:
        '''
        if self.model_name == "inception_v3":
            net = models.inception_v3(pretrained=True)
        elif self.model_name == "densenet121":
            net = models.densenet121(pretrained=True)

        # 关闭求导，节约大量的显存
        for param in net.parameters():
            param.requires_grad = False

        # 去掉Top层
        base_model = list(net.children())[:-1]
        extractor = nn.Sequential(*base_model)
        return extractor

    def extract_features_save_to_file(self, samples_name, batch_size):
        '''
        提取图样本集的特征向量，并存盘
        :param samples_name: 样本集的文件列表文件
        :param batch_size: 每批的图片数量
        :return: 特征向量的存盘文件
        '''

        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_data = Image_Dataset(Xtrain, Ytrain)

        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_data = Image_Dataset(Xtest, Ytest)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)

        model = self.load_pretrained_model()
        print(model)

        if self.use_GPU:
            model.cuda()

        model.eval()

        for stage in ["train", "test"]:
            if stage == "train":
                data_loader = train_loader
            else:
                data_loader = test_loader

            data_len = data_loader.__len__()
            features = []
            for step, (x, y) in enumerate(data_loader):
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                else:
                    b_x = Variable(x)  # batch x

                output = model(b_x)
                f = output.cpu().data.numpy()
                avg_f = np.mean(f, axis=(-2, -1)) # 全局平均池化
                features.extend(avg_f)
                print('extracting features => %d / %d ' % (step + 1, data_len))

            if stage == "train":
                Y = Ytrain
            else:
                Y = Ytest

            save_path = "{}/data/pytorch/{}_{}_{}".format(self._params.PROJECT_ROOT, self.model_name, samples_name, stage)

            labels = Y[:len(features)]
            np.savez(save_path + "_features", features, labels)

    
    #################################################################################################
    #              SVM
    ##################################################################################################

    def train_top_svm(self, train_filename, test_filename):
        '''
        训练SVM分类器，作为CNN网络的新TOP。 包含参数寻优过程
        :param train_filename: 保存特征的训练集文件
        :param test_filename: 测试集文件
        :return:
        '''
        if (not self.model_name in train_filename) or (not self.model_name in test_filename) \
                or (not self.patch_type in train_filename) or (not self.patch_type in test_filename):
            return

        data_path = "{}/data/pytorch/{}".format(self._params.PROJECT_ROOT, test_filename)
        D = np.load(data_path)
        test_features = D['arr_0']
        test_label = D['arr_1']

        data_path = "{}/data/pytorch/{}".format(self._params.PROJECT_ROOT, train_filename)
        D = np.load(data_path)
        train_features = D['arr_0']
        train_label = D['arr_1']

        max_iter = 500
        # model_params = [ {'C':0.0001}, {'C':0.001 }, {'C':0.01}, {'C':0.1},
        #                  {'C':0.5}, {'C':1.0}, {'C':1.2}, {'C':1.5},
        #                  {'C':2.0}, {'C':10.0} ]
        model_params = [{'C': 0.01}]
        # K_num = [100, 200, 300, 500, 1024, 2048]
        #
        result = {'pred': None, 'score': 0, 'clf': None}
        # for item in K_num:
        #     sb = SelectKBest(k=item).fit(train_features, train_label)
        #     train_x_new = sb.transform(train_features)
        #     test_x_new = sb.transform(test_features)

        # 进行了简单的特征选择，选择全部特征。
        # inception_v3 ： the best score = 0.8891836734693878, k = 2048， C=0.0001
        # densenet121: the best score = 0.9151020408163265, k=1024, C=0.01
        # resnet50: the best score = 0.8187755102040817, C=0.5
        feature_num = len(train_features[0])
        for params in model_params:
            clf = LinearSVC(**params, max_iter=max_iter, verbose=0)
            clf.fit(train_features, train_label)
            y_pred = clf.predict(test_features)
            score = metrics.accuracy_score(test_label, y_pred)
            print('feature num = {}, C={:8f} => score={:5f}'.format(feature_num, params['C'], score))

            if score > result["score"]:
                result = {'pred': y_pred, 'score': score, 'clf': clf}

        print("the best score = {}".format(result["score"]))

        print("Classification report for classifier %s:\n%s\n"
              % (result["clf"], metrics.classification_report(test_label, result["pred"])))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, result["pred"]))

        model_file = self._params.PROJECT_ROOT + "/models/pytorch/svm_{}_{}.model".format(self.model_name, self.patch_type)
        joblib.dump(result["clf"], model_file)
        return