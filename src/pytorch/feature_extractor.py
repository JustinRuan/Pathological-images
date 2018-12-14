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

class Feature_Extractor(object):
    def __init__(self, params, model_name, patch_type):

        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        self.use_GPU = True

    def load_pretrained_model(self):
        if self.model_name == "inception_v3":
            net = models.inception_v3(pretrained=True)
        elif self.model_name == "densenet121":
            net = models.densenet121(pretrained=True)

        # 关闭求导，节约大量的显存
        for param in net.parameters():
            param.requires_grad = False

        base_model = list(net.children())[:-1]
        extractor = nn.Sequential(*base_model)
        return extractor

    def extract_features_save_to_file(self, samples_name, batch_size):

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

    
