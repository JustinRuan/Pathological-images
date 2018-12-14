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
from core.util import latest_checkpoint
from pytorch.net import Simple_CNN
from pytorch.net import DenseNet
from core.util import read_csv_file
from pytorch.image_dataset import Image_Dataset

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
        # if (not os.path.exists(self.model_root)):
        #     os.makedirs(self.model_root)
        self.use_GPU = True

    def create_densenet(self, depth):
        # Get densenet configuration
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]

        if self.patch_type in ["cifar10", "cifar100"]:  # 32x32
            # Models
            model = DenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                small_inputs=True, # 32 x 32的图片为True
                avgpool_size=8,
                efficient=True,
            )
        elif self.patch_type == "500_128": # 128 x 128
            # Models
            model = DenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                small_inputs=False, # 32 x 32的图片为True
                avgpool_size=7,
                efficient=True,
            )
        elif self.patch_type in ["2000_256", "4000_256"]: # 256 x 256
            # Models
            model = DenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                small_inputs=False, # 32 x 32的图片为True
                avgpool_size=14,
                efficient=True,
            )
        return  model

    def create_initial_model(self):

        if self.model_name == "simple_cnn":
            model = Simple_CNN(self.num_classes, self.image_size)
        elif self.model_name == "densenet_22":
            model = self.create_densenet(depth=22)

        return model

    def load_model(self, model_file = None):

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            model = torch.load(model_file)
            return model
        else:
            checkpoint_dir = self.model_root
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                model = torch.load(latest)
            else:
                model = self.create_initial_model()
            return model

    def train_model(self, samples_name = None, batch_size=100, epochs=20):
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        elif self.patch_type in ["500_128", "2000_256", "4000_256"]:
            train_data, test_data = self.load_custom_data(samples_name)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        print(model)
        if self.use_GPU:
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #学习率为0.01的学习器
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
        loss_func = nn.CrossEntropyLoss()

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()

            train_data_len = train_data.__len__() // batch_size + 1
            total_loss = 0
            for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                    b_y = Variable(y).cuda()  # batch y
                else:
                    b_x = Variable(x)  # batch x
                    b_y = Variable(y)  # batch y

                output = model(b_x)  # cnn output
                loss = loss_func(output, b_y)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                # 数据统计
                _, preds = torch.max(output, 1)

                running_loss = loss.item()
                running_corrects = torch.sum(preds == b_y.data)
                total_loss += running_loss
                print('%d / %d ==> Loss: %.4f | Acc: %.4f '
                      % (step, train_data_len, running_loss, running_corrects.double()/b_x.size(0)))

            scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            model.eval()
            for x, y in test_loader:
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                    b_y = Variable(y).cuda()  # batch y
                else:
                    b_x = Variable(x)  # batch x
                    b_y = Variable(y)  # batch y

                output = model(b_x)
                loss = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects += torch.sum(preds == b_y.data)

            test_data_len = test_data.__len__()
            epoch_loss=running_loss / test_data_len
            epoch_acc=running_corrects.double() / test_data_len

            torch.save(model, self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.h5".format(epoch+1, epoch_loss, epoch_acc))

        return

    def load_cifar_data(self, patch_type):
        data_root = os.path.join(os.path.expanduser('~'), '.keras/datasets/') # 共用Keras下载的数据

        if patch_type == "cifar10":
            train_data = torchvision.datasets.cifar.CIFAR10(
                root=data_root,  # 保存或者提取位置
                train=True,  # this is training data
                transform=torchvision.transforms.ToTensor(),
                download = False
            )
            test_data = torchvision.datasets.cifar.CIFAR10(root=data_root, train=False,
                                                   transform=torchvision.transforms.ToTensor())
            return train_data, test_data

    def load_custom_data(self, samples_name):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_data = Image_Dataset(Xtrain, Ytrain)

        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_data = Image_Dataset(Xtest, Ytest)
        return  train_data, test_data



