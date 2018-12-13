#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块

from pytorch.net import Simple_CNN

class CNN_Classifier(object):

    def __init__(self, params, model_name, patch_type):

        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        if self.patch_type == "500_128":
            self.num_classes = 2
            self.image_size = 128
        elif self.patch_type in ["2000_256", "4000_256"]:
            self.num_classes = 2
            self.image_size = 256
        elif self.patch_type == "cifar10":
            self.num_classes = 10
            self.image_size = 32
        elif self.patch_type == "cifar100":
            self.num_classes = 100
            self.image_size = 32

        self.model_root = "{}/models/pytorch/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)

    def create_initial_model(self):

        if self.model_name == "simple_cnn":
            model = Simple_CNN(self.num_classes, self.image_size)

        return model

    def load_model(self, model_file = None):

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            model = None
            return model
        else:
            model = self.create_initial_model()
            return model

    def train_model_cifar(self, batch_size=100, epochs=20):
        data_root = os.path.join(os.path.expanduser('~'), '.keras/datasets/') # 共用Keras下载的数据

        if self.patch_type == "cifar10":
            train_data = torchvision.datasets.cifar.CIFAR10(
                root=data_root,  # 保存或者提取位置
                train=True,  # this is training data
                transform=torchvision.transforms.ToTensor(),
                download = False
            )
            test_data = torchvision.datasets.cifar.CIFAR10(root=data_root, train=False,
                                                   transform=torchvision.transforms.ToTensor())

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            running_loss = 0
            running_corrects = 0
            total = 0
            train_data_len = len(train_data.train_data) // batch_size + 1

            for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x = Variable(x)  # batch x
                b_y = Variable(y)  # batch y

                output = model(b_x)  # cnn output
                loss = loss_func(output, b_y)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                # 数据统计
                _, preds = torch.max(output, 1)
                # 注意如果你想统计loss，切勿直接使用loss相加，而是使用loss.data[0]。因为loss是计算图的一部分，
                # 如果你直接加loss，代表total loss同样属于模型一部分，那么图就越来越大
                running_loss += loss.item() * b_x.size(0)
                total += b_y.size(0)
                running_corrects += torch.sum(preds == b_y.data)
                print('%d / %d ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (step, train_data_len, running_loss/(step+1), (100 * running_corrects.double())/total,
                         running_corrects, total))

            running_loss=0.0
            running_corrects=0
            model.eval()
            for x, y in test_loader:
                b_x = Variable(x)  # batch x
                b_y = Variable(y)  # batch y
                output = model(b_x)
                loss = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects += torch.sum(preds == b_y.data)

            test_data_len = len(test_data.test_data)
            epoch_loss=running_loss / test_data_len
            epoch_acc=running_corrects.double() / test_data_len

            torch.save(model, self.model_root + "/cp_{}_{}_{}.h5".format(epoch, epoch_loss, epoch_acc))

        return



