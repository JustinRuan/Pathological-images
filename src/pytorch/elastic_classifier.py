#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-05-31'

"""

import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
from sklearn import metrics
from torch.autograd import Variable
from torchsummary import summary

from core.util import latest_checkpoint
from core.util import read_csv_file
from pytorch.image_dataset import Image_Dataset, Image_Dataset_MSC
from pytorch.net import DenseNet, SEDenseNet
from pytorch.net import Simple_CNN
from pytorch.util import get_image_blocks_itor, get_image_blocks_msc_itor, get_image_blocks_batch_normalize_itor, \
    get_image_file_batch_normalize_itor
import datetime
from core import Block
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import models
from scipy import stats
from pytorch.loss_function import CenterLoss

from pytorch.cnn_classifier import Simple_Classifier

class Shadow_Classifier(nn.Module):

    def __init__(self, initial_params, device):
        super(Shadow_Classifier, self).__init__()

        self.device = device

        num_classes, in_features = initial_params.size()
        self.classifier = nn.Linear(in_features, num_classes)

        # Initialization
        self.classifier.weight = initial_params

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.classifier(x)
        return out

    def train_myself(self, train_features, train_label, weight, batch_size, loss_weight):

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_features),
                                                    torch.from_numpy(train_label).long(),
                                                    torch.from_numpy(weight).float())
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        classifi_loss= nn.CrossEntropyLoss(reduction='none')
        center_loss = CenterLoss(2, 2)

        self.to(self.device)
        classifi_loss.to(self.device)
        center_loss.to(self.device)

        classifi_optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimzer4center = torch.optim.SGD(center_loss.parameters(), lr=0.1)
        epochs = 5
        self.train()
        for epoch in range(epochs):
            for step, (x, y, w) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x = Variable(x.to(self.device))  # batch x
                b_y = Variable(y.to(self.device))  # batch y
                b_w = w.to(self.device)

                output = self.forward(b_x)  # cnn output is features, not logits
                # cross entropy loss + center loss
                cross_Loss = (b_w * classifi_loss(output, b_y)).sum()
                loss = cross_Loss + loss_weight * center_loss(b_y, output, b_w)

                classifi_optimizer.zero_grad()  # clear gradients for this training step
                optimzer4center.zero_grad()
                loss.backward()  # backpropagation, compute gradients
                classifi_optimizer.step()
                optimzer4center.step()

    def predict(self, train_features, batch_size):
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(train_features),
                                                    torch.zeros(len(train_features)))
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        probability = []
        prediction = []
        low_dim_features = []

        for step, (x, _) in enumerate(test_loader):
            b_x = Variable(x.to(self.device))  # batch x

            output = self.forward(b_x)
            output_softmax = nn.functional.softmax(output, dim=1)
            probs, preds = torch.max(output_softmax, 1)

            low_dim_features.extend(output.detach().cpu().numpy())
            probability.extend(probs.detach().cpu().numpy())
            prediction.extend(preds.detach().cpu().numpy())

        return probability, prediction, low_dim_features


class Elastic_Classifier(Simple_Classifier):
    def __init__(self, params, model_name, patch_type, **kwargs):
        super(Elastic_Classifier, self).__init__(params, model_name, patch_type,**kwargs)
        self.shadow_classifier = None
        return

    def construct_shadow_classifier(self, model):
        params=model.classifier.weight
        self.shadow_classifier = Shadow_Classifier(params, device=self.device)
        return

    def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return: 预测结果与概率
        '''

        seeds_itor = get_image_blocks_itor(src_img, scale, seeds, patch_size, patch_size, batch_size,
                                           normalization=self.normal_func)

        if self.model is None:
            self.model = self.load_pretrained_model_on_predict()
            self.construct_shadow_classifier(self.model)

            self.model.to(self.device)
            self.model.eval()

        len_seeds = len(seeds)
        data_len = len(seeds) // batch_size
        if len_seeds % batch_size > 0:
            data_len += 1

        probability = []
        prediction = []
        high_dim_features = []
        low_dim_features = []
        for step, x in enumerate(seeds_itor):
            b_x = Variable(x.to(self.device))

            output = self.model(b_x) # model最后不包括一个softmax层
            output_softmax = nn.functional.softmax(output, dim =1)
            probs, preds = torch.max(output_softmax, 1)

            high_dim_features.extend(self.model.out_feature.cpu().numpy())
            low_dim_features.extend(output.detach().cpu().numpy())
            probability.extend(probs.detach().cpu().numpy())
            prediction.extend(preds.detach().cpu().numpy())
            print('predicting => %d / %d ' % (step + 1, data_len))

        low_dim_features = np.array(low_dim_features)
        prediction = np.array(prediction)
        probability = np.array(probability)

        if len(prediction)> 6 and np.sum(prediction) > 3: # 至少3个Cancer样本输入
            # 对样本进行加权处理
            weight = self.correct_sample_weights(low_dim_features, prediction)

            high_dim_features = np.array(high_dim_features)
            prediction = np.array(prediction)
            # 开启弹性调整过程
            self.shadow_classifier.train_myself(high_dim_features, prediction, weight, batch_size // 2, 0.001)
            self.shadow_classifier.predict(high_dim_features, batch_size)

        return probability, prediction, low_dim_features #new_features #low_dim_features

    def correct_sample_weights(self, features, prediction):
        features_0 = features[:,0]
        feat = features_0[prediction == 1]
        cancer_count = len(feat)
        if cancer_count > 3:
            cancer_mean = np.mean(feat)
            cancer_std = np.std(feat)
        else:
            # cancer_mean = -1.5
            # cancer_std = 1.0
            raise ValueError("cancer count is less than 3!")

        feat = features_0[prediction == 0]
        normal_mean = np.mean(feat)
        normal_std = np.std(feat)

        if cancer_count > 3 :
            cancer_interval = stats.t.interval(0.95, cancer_count - 1, cancer_mean, cancer_std)
            normal_interval = stats.norm.interval(0.95, loc=normal_mean, scale=normal_std)

            print(" cancer intervel: {:.4f} {:.4f}, ".format(cancer_interval[0], cancer_interval[1]),
                  "normal interval: {:.4f} {:.4f}".format(normal_interval[0], normal_interval[1]))
        # else:
        #     cancer_interval = [-3, 0]
        #     normal_interval = stats.norm.interval(0.95, loc=normal_mean, scale=normal_std)
        #     print("normal interval: {:.4f} {:.4f}".format(normal_interval[0], normal_interval[1]))

        normal_edge = normal_interval[0]
        cancer_edge = cancer_interval[1]

        S = 0.01
        weight = S * np.ones(prediction.shape, dtype=np.float)
        if normal_edge > cancer_edge:
            # 两个类中心完全分离,
            return np.ones(prediction.shape, dtype=np.float)
        elif normal_edge + normal_std > cancer_edge - cancer_std:
            # 发生的重叠情况
            normal_edge = normal_edge + normal_std
            cancer_edge = cancer_edge - cancer_std
        else:
            # 严重重叠
            normal_edge = normal_mean
            cancer_edge = cancer_mean

        weight[features_0 > normal_edge] = 1.0
        weight[features_0 < cancer_edge] = 1.0
        print(">>>> suppress some suspicious : ", np.sum(weight != 1))

        return weight

    # def correct(self, features, prediction):
    #     features_0 = features[:,0]
    #     feat = features_0[prediction == 1]
    #     cancer_mean = np.mean(feat)
    #     cancer_std = np.std(feat)
    #     cancer_count = len(feat)
    #
    #     feat = features_0[prediction == 0]
    #     normal_mean = np.mean(feat)
    #     normal_std = np.std(feat)
    #
    #     if cancer_count > 3 :
    #         cancer_interval = stats.t.interval(0.95, cancer_count - 1, cancer_mean, cancer_std)
    #         normal_interval = stats.norm.interval(0.95, loc=normal_mean, scale=normal_std)
    #
    #         print(" cancer intervel: {:.4f} {:.4f}, ".format(cancer_interval[0], cancer_interval[1]),
    #               "normal interval: {:.4f} {:.4f}".format(normal_interval[0], normal_interval[1]))
    #     else:
    #         cancer_interval = [-3, 0]
    #         cancer_mean = -1.5
    #         cancer_std = 1.0
    #         normal_interval = stats.norm.interval(0.95, loc=normal_mean, scale=normal_std)
    #         print("normal interval: {:.4f} {:.4f}".format(normal_interval[0], normal_interval[1]))
    #
    #     normal_edge = normal_interval[0]
    #     cancer_edge = cancer_interval[1]
    #
    #     tag = np.ones(prediction.shape, dtype=np.bool)
    #     if normal_edge > cancer_edge:
    #         # 两个类中心完全分离,
    #         return features
    #     elif normal_edge + normal_std > cancer_edge - cancer_std:
    #         # 发生的重叠情况
    #         normal_edge = normal_edge + normal_std
    #         cancer_edge = cancer_edge - cancer_std
    #     else:
    #         # 严重重叠
    #         normal_edge = normal_mean
    #         cancer_edge = cancer_mean
    #
    #     tag[features_0 > normal_edge] = False
    #     tag[features_0 < cancer_edge] = False
    #     features[tag] = np.array([None, None])
    #     print(">>>> Exclude some suspicious : ", np.sum(tag == False))
    #
    #     return features
