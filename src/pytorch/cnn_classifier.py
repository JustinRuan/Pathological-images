#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
from core.util import latest_checkpoint
from pytorch.net import Simple_CNN
from pytorch.net import DenseNet
from pytorch.net import SEDenseNet
from core.util import read_csv_file
from pytorch.image_dataset import Image_Dataset
from pytorch.image_dataset2 import Image_Dataset2
from pytorch.util import get_image_blocks_itor

class CNN_Classifier(object):

    def __init__(self, params, model_name, patch_type,MultiTask=False):
        '''
        初始化
        :param params: 系统参数
        :param model_name: 分类器算法的代号
        :param patch_type: 分类器处理的图块类型的代号
        '''
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS
        self.MultiTask=MultiTask
        if self.patch_type == "500_128":
            self.num_classes = 2
            self.image_size = 128
        elif self.patch_type in ["2000_256", "4000_256","500_128+2000_256+4000_256"]:
            self.num_classes = 2
            self.image_size = 256
        elif self.patch_type == "cifar10":
            self.num_classes = 10
            self.image_size = 32
        elif self.patch_type == "cifar100":
            self.num_classes = 100
            self.image_size = 32
        if MultiTask:
            self.model_root = "{}/models/pytorch/{}_{}_MultiTask".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        else:
            self.model_root = "{}/models/pytorch/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        self.use_GPU = True

    def create_densenet(self, depth):
        '''
        生成指定深度的Densenet
        :param depth: 深度
        :return: 网络模型
        '''
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
                MultiTask=self.MultiTask
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
                MultiTask=self.MultiTask
            )
        elif self.patch_type in ["2000_256", "4000_256","500_128+2000_256+4000_256"]: # 256 x 256
            # Models
            model = DenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                small_inputs=False, # 32 x 32的图片为True
                avgpool_size=14,
                efficient=True,
                MultiTask=self.MultiTask
            )
        return  model
    def create_sedensenet(self, depth):
        # Get densenet configuration
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]

        if self.patch_type in ["cifar10", "cifar100"]:  # 32x32
            # Models
            model = SEDenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                avgpool_size=8,
                MultiTask=self.MultiTask
            )
        elif self.patch_type == "500_128": # 128 x 128
            # Models
            model = SEDenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                avgpool_size=7,
                MultiTask=self.MultiTask
            )
        elif self.patch_type in ["2000_256", "4000_256","500_128+2000_256+4000_256"]: # 256 x 256
            # Models
            model = SEDenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                avgpool_size=14,
                MultiTask=self.MultiTask
            )
        return  model

    def create_initial_model(self):
        '''
        生成初始化的模型
        :return:网络模型
        '''
        if self.model_name == "simple_cnn":
            model = Simple_CNN(self.num_classes, self.image_size)
        elif self.model_name == "densenet_22":
            model = self.create_densenet(depth=22)
        elif self.model_name =="densenet_40":
            model=self.create_densenet(depth=40)
        elif self.model_name == "sedensenet_22":
            model = self.create_sedensenet(depth=22)
        elif self.model_name =="sedensenet_40":
            model=self.create_sedensenet(depth=40)

        return model

    def load_model(self, model_file = None):
        '''
        加载模型
        :param model_file: 模型文件
        :return: 网络模型
        '''
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
        '''
        训练模型
        :param samples_name: 自制训练集的代号
        :param batch_size: 每批的图片数量
        :param epochs:epoch数量
        :return:
        '''
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        elif self.patch_type in ["500_128", "2000_256", "4000_256"]:
            train_data, test_data = self.load_custom_data(samples_name)
        elif self.patch_type in ["500_128+2000_256+4000_256"]:
            train_data, test_data = self.load_Multicustom_data(samples_name)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        # train_gain_loader = Data.DataLoader(dataset=train_gain_data, batch_size=batch_size, shuffle=True)
        # test_gain_loader = Data.DataLoader(dataset=test_gain_data, batch_size=batch_size, shuffle=False)
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
            # 开始训练
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
            # 开始评估
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

            torch.save(model, self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch+1, epoch_loss, epoch_acc))

        return

    #多任务下的训练过程
    def train_model_MultiTask(self, samples_name=None, samples_name2=None, batch_size=100, epochs=20):
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        elif self.patch_type in ["500_128", "2000_256", "4000_256"]:
            train_data, test_data = self.load_custom_data(samples_name)
        # elif self.patch_type in ["500_128+2000_256+4000_256"]:
        #     train_data, test_data, train_gain_data, test_gain_data = self.load_Multicustom_data(samples_name)
        elif self.patch_type in ["500_128+2000_256+4000_256"]:
            train_data, test_data = self.load_Multicustom_data2(samples_name)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        model = self.load_model(model_file=None)
        print(model)
        if self.use_GPU:
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
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
                    b_y = Variable(y[0]).cuda()  # batch y
                    b_z = Variable(y[1]).cuda()  # batch z
                else:
                    b_x = Variable(x)  # batch x
                    b_y = Variable(y[0])  # batch y
                    b_z = Variable(y[1])  # batch z
                output1 = model(b_x)[0]  # cnn output
                output2 = model(b_x)[1]  # cnn output
                loss1 = loss_func(output1, b_y)  # cross entropy loss
                loss2 = loss_func(output2, b_z)  # cross entropy loss
                loss=0.95*loss1+0.05*loss2
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                # 数据统计
                _, preds1 = torch.max(output1, 1)
                _, preds2 = torch.max(output2, 1)
                running_loss = loss.item()
                running_corrects1 = torch.sum(preds1 == b_y.data)
                running_corrects2 = torch.sum(preds2 == b_z.data)
                total_loss += running_loss
                print('%d / %d ==> Loss: %.4f | Acc: %.4f | Acc_gain: %.4f '
                      % (step, train_data_len, running_loss, running_corrects1.double() / b_x.size(0), running_corrects2.double() / b_x.size(0)))

            scheduler.step(total_loss)

            running_loss = 0.0
            running_corrects1 = 0
            running_corrects2 = 0
            model.eval()
            for x, y  in test_loader:
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                    b_y = Variable(y[0]).cuda()  # batch y
                    b_z = Variable(y[1]).cuda()  # batch z
                else:
                    b_x = Variable(x)  # batch x
                    b_y = Variable(y[0])  # batch y
                    b_z = Variable(y[1])  # batch z
                output1 = model(b_x)[0]
                output2 = model(b_x)[1]

                loss1 = loss_func(output1, b_y)  # cross entropy loss
                loss2 = loss_func(output2, b_z)  # cross entropy loss
                loss = 0.95 * loss1 + 0.05 * loss2


                _, preds1 = torch.max(output1, 1)
                _, preds2 = torch.max(output2, 1)
                running_loss += loss.item() * b_x.size(0)

                running_corrects1 += torch.sum(preds1 == b_y.data)
                running_corrects2 += torch.sum(preds2 == b_z.data)

                # output2 = model(b_x)
                # loss = loss_func(output2, b_z)
                #
                # _, preds = torch.max(output2, 1)
                # running_loss += loss.item() * b_x.size(0)
                # running_corrects += torch.sum(preds == b_z.data)
            test_data_len = test_data.__len__()
            epoch_loss = running_loss / test_data_len
            epoch_acc = running_corrects1.double() / test_data_len
            epoch_acc_gain = running_corrects2.double() / test_data_len
            torch.save(model, self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}-{:.4f}.h5".format(epoch + 1, epoch_loss, epoch_acc,epoch_acc_gain))

        return

    def load_cifar_data(self, patch_type):
        '''
        加载cifar数量
        :param patch_type: cifar 数据的代号
        :return:
        '''
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
    #加载三个倍镜下的所有图片
    def load_Multicustom_data(self, samples_name):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        if(samples_name=="T_NC_500_128+2000_256+4000_256"):
            #获取三个倍镜下的所有训练测试数据
            train_list_5 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_500_128")
            test_list_5 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_500_128")

            train_list_20 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_2000_256")
            test_list_20 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_2000_256")

            train_list_40 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_4000_256")
            test_list_40 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_4000_256")
           #分别生成三个倍镜下的训练数据
            Xtrain_5, Ytrain_5 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_5)

            Xtrain_20, Ytrain_20 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_20)

            Xtrain_40, Ytrain_40 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_40)
            #将三个倍镜下的训练数据合并
            Xtrain = Xtrain_5 + Xtrain_20 + Xtrain_40
            Ytrain = Ytrain_5 + Ytrain_20 + Ytrain_40
            train_data = Image_Dataset(Xtrain, Ytrain)
            # 分别生成三个倍镜下的测试数据
            Xtest_5, Ytest_5 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_5)

            Xtest_20, Ytest_20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_20)

            Xtest_40, Ytest_40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_40)
            # 将三个倍镜下的测试数据合并
            Xtest = Xtest_5 + Xtest_20 + Xtest_40
            Ytest = Ytest_5 + Ytest_20 + Ytest_40
            test_data = Image_Dataset(Xtest, Ytest)

            return train_data, test_data

    #多任务训练时，加载三个倍镜下的所有图片以及三种不同倍镜的标注信息
    def load_Multicustom_data2(self, samples_name):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        if (samples_name == "T_NC_500_128+2000_256+4000_256"):
            # 获取三个倍镜下的所有训练测试数据
            train_list_5 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_500_128")
            test_list_5 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_500_128")

            train_list_20 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_2000_256")
            test_list_20 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_2000_256")

            train_list_40 = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_4000_256")
            test_list_40 = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, "T_NC_4000_256")
            # 分别生成三个倍镜下的训练数据，每个倍镜下包含有癌无癌和倍镜两个标注信息，5倍镜-0,20倍镜-1,40倍镜-2
            Xtrain_5, Ytrain_5 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_5)
            Ytrain_gain_5 = np.zeros(len(Ytrain_5)).astype("int64").tolist()

            Xtrain_20, Ytrain_20 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_20)
            Ytrain_gain_20 = np.ones(len(Ytrain_20)).astype("int64").tolist()

            Xtrain_40, Ytrain_40 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list_40)
            Ytrain_gain_40 = (np.ones(len(Ytrain_40)) + 1).astype("int64").tolist()
            # 将三个倍镜下的训练数据合并
            Xtrain = Xtrain_5 + Xtrain_20 + Xtrain_40
            Ytrain = Ytrain_5 + Ytrain_20 + Ytrain_40
            Ytrain_gain = Ytrain_gain_5 + Ytrain_gain_20 + Ytrain_gain_40
            train_data = Image_Dataset2(Xtrain, Ytrain, Ytrain_gain)
            # 分别生成三个倍镜下的测试数据，每个倍镜下包含有癌无癌和倍镜两个标注信息，5倍镜-0,20倍镜-1,40倍镜-2
            Xtest_5, Ytest_5 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_5)
            Ytest_gain_5 = np.zeros(len(Ytest_5)).astype("int64").tolist()

            Xtest_20, Ytest_20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_20)
            Ytest_gain_20 = np.ones(len(Ytest_20)).astype("int64").tolist()

            Xtest_40, Ytest_40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list_40)
            Ytest_gain_40 = (np.ones(len(Ytest_40)) + 1).astype("int64").tolist()

            # 将三个倍镜下的测试数据合并
            Xtest = Xtest_5 + Xtest_20 + Xtest_40
            Ytest = Ytest_5 + Ytest_20 + Ytest_40
            Ytest_gain = Ytest_gain_5 + Ytest_gain_20 + Ytest_gain_40
            test_data = Image_Dataset2(Xtest, Ytest, Ytest_gain)

            return train_data, test_data
    #联合训练时，分别加载单个倍镜下的测试集进行单独测试，其中参数MultiTask用来标记是否是多任务
    def load_test_data(self, samples_name,MultiTask):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        if MultiTask:
            if samples_name=="T_NC_500_128":
                Ytest_gain = np.zeros(len(Ytest)).astype("int64").tolist()
            elif samples_name=="T_NC_2000_256":
                Ytest_gain = np.ones(len(Ytest)).astype("int64").tolist()
            elif samples_name=="T_NC_4000_256":
                Ytest_gain = (np.ones(len(Ytest))+1).astype("int64").tolist()
            test_data = Image_Dataset2(Xtest, Ytest, Ytest_gain)

        else:
            test_data = Image_Dataset(Xtest, Ytest)

        return  test_data

    def predict_on_samples(self,patch_type,batch_size=128,MultiTask=False,model_file=None):
        test_data=self.load_test_data(samples_name="T_NC_{}".format(patch_type),MultiTask=MultiTask)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        model = self.load_model(model_file=model_file)
        loss_func = nn.CrossEntropyLoss()
        running_loss = 0.0
        running_corrects = 0
        running_corrects1 = 0
        running_corrects2 = 0
        if MultiTask:
            for x, y  in test_loader:
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                    b_y = Variable(y[0]).cuda()  # batch y
                    b_z = Variable(y[1]).cuda()  # batch z
                else:
                    b_x = Variable(x)  # batch x
                    b_y = Variable(y[0])  # batch y
                    b_z = Variable(y[1])  # batch z
                output1 = model(b_x)[0]
                output2 = model(b_x)[1]

                loss1 = loss_func(output1, b_y)  # cross entropy loss
                loss2 = loss_func(output2, b_z)  # cross entropy loss
                loss = 0.95 * loss1 + 0.05 * loss2


                _, preds1 = torch.max(output1, 1)
                _, preds2 = torch.max(output2, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects1 += torch.sum(preds1 == b_y.data)
                running_corrects2 += torch.sum(preds2 == b_z.data)

                # output2 = model(b_x)
                # loss = loss_func(output2, b_z)
                #
                # _, preds = torch.max(output2, 1)
                # running_loss += loss.item() * b_x.size(0)
                # running_corrects += torch.sum(preds == b_z.data)
            test_data_len = test_data.__len__()
            epoch_loss = running_loss / test_data_len
            epoch_acc = running_corrects1.double() / test_data_len
            epoch_acc_gain = running_corrects2.double() / test_data_len
            return epoch_loss,epoch_acc,epoch_acc_gain
        else:
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
            epoch_loss = running_loss / test_data_len
            epoch_acc = running_corrects.double() / test_data_len
            return epoch_loss, epoch_acc


    def load_pretrained_model_on_predict(self, patch_type):
        '''
        加载已经训练好的存盘网络文件
        :param patch_type: 分类器处理图块的类型
        :return: 网络模型
        '''
        net_file = {"500_128":  "densenet_22_500_128_cp-0017-0.2167-0.9388.pth",
                    "2000_256": "densenet_22_2000_256-cp-0019-0.0681-0.9762.pth",
                    "4000_256": "densenet_22_4000_256-cp-0019-0.1793-0.9353.pth", }

        model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[patch_type])
        model = self.load_model(model_file=model_file)

        # 关闭求导，节约大量的显存
        for param in model.parameters():
            param.requires_grad = False
        return model

    def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return: 预测结果与概率
        '''
        seeds_itor = get_image_blocks_itor(src_img, scale, seeds, patch_size, patch_size, batch_size)

        model = self.load_pretrained_model_on_predict(self.patch_type)
        # print(model)

        if self.use_GPU:
            model.cuda()
        model.eval()

        data_len = len(seeds) // batch_size + 1
        results = []

        for step, x in enumerate(seeds_itor):
            if self.use_GPU:
                b_x = Variable(x).cuda()  # batch x
            else:
                b_x = Variable(x)  # batch x

            output = model(b_x)
            output_softmax = nn.functional.softmax(output)
            probs, preds = torch.max(output_softmax, 1)
            for prob, pred in zip(probs.cpu().numpy(), preds.cpu().numpy()):
                results.append((pred, prob))
            print('predicting => %d / %d ' % (step + 1, data_len))

        return results


