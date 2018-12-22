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
from pytorch.util import get_image_blocks_itor

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn import metrics
from collections import OrderedDict
from core.util import latest_checkpoint


NUM_CLASSES = 2

class Transfer(object):
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

        self.model_root = "{}/models/pytorch/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)

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

    def save_pretrained_base_model(self):
        base_model = self.load_pretrained_model()

        model_file = "{}/models/pytorch/trained/{}_base_model.pth".format(self._params.PROJECT_ROOT, self.model_name)
        torch.save(base_model, model_file)

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


    def extract_feature(self, src_img, scale, patch_size, seeds, batch_size):
        seeds_itor = get_image_blocks_itor(src_img, scale, seeds, patch_size, patch_size, batch_size)

        model_file = "{}/models/pytorch/trained/{}_base_model.pth".format(self._params.PROJECT_ROOT, self.model_name)
        base_model = torch.load(model_file)

        print(base_model)

        if self.use_GPU:
            base_model.cuda()
        base_model.eval()

        data_len = len(seeds) // batch_size + 1
        features = []

        for step, x in enumerate(seeds_itor):
            if self.use_GPU:
                b_x = Variable(x).cuda()  # batch x
            else:
                b_x = Variable(x)  # batch x

            output = base_model(b_x)
            f = output.cpu().data.numpy()
            avg_f = np.mean(f, axis=(-2, -1))  # 全局平均池化
            features.extend(avg_f)
            print('extracting features => %d / %d ' % (step + 1, data_len))

        return features

    #################################################################################################
    #              NN
    ##################################################################################################

    def create_new_top_cnn_model(self):
        features_num = {"inception_v3": 2048,
                        "densenet121": 1024,
                        "densenet169": 1664,
                        "densenet201": 1920,
                        "resnet50": 2048,
                        "inception_resnet_v2": 1536,
                        "vgg16": 512,
                        "mobilenet_v2": 1280}

        top_model = nn.Sequential(OrderedDict([
            ('top_Dense', nn.Linear(features_num[self.model_name], 1024)),
            ('top_relu', nn.ReLU()),
            ('predictions', nn.Linear(1024, NUM_CLASSES)),
            ('top_softmax', nn.Softmax())
        ]))

        return top_model

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
                model = self.create_new_top_cnn_model()
            return model

    def train_top_cnn_model(self, train_filename, test_filename, batch_size, epochs):
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

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_features),
                                                    torch.from_numpy(train_label).long())
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_features),
                                                    torch.from_numpy(test_label).long())
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)

        # if (not os.path.exists(self.model_root)):
        #     os.makedirs(self.model_root)

        model = self.load_model()
        print(model)

        if self.use_GPU:
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
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
                      % (step, train_data_len, running_loss, running_corrects.double() / b_x.size(0)))

            scheduler.step(total_loss)

            running_loss = 0.0
            running_corrects = 0
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
            epoch_loss = running_loss / test_data_len
            epoch_acc = running_corrects.double() / test_data_len

            torch.save(model, self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss, epoch_acc))

        return



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
        model_params = [ {'C':0.0001}, {'C':0.001 }, {'C':0.01}, {'C':0.1},
                         {'C':0.5}, {'C':1.0}, {'C':1.2}, {'C':1.5},
                         {'C':2.0}, {'C':10.0} ]
        # model_params = [{'C': 0.01}]
        # K_num = [100, 200, 300, 500, 1024, 2048]
        #
        result = {'pred': None, 'score': 0, 'clf': None}
        # for item in K_num:
        #     sb = SelectKBest(k=item).fit(train_features, train_label)
        #     train_x_new = sb.transform(train_features)
        #     test_x_new = sb.transform(test_features)

        # 进行了简单的特征选择，选择全部特征。
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

        model_file = self._params.PROJECT_ROOT + "/models/pytorch/svm_{}_{}_{:.4f}.model".format(self.model_name,
                                                                                                 self.patch_type,
                                                                                                 result["score"])
        joblib.dump(result["clf"], model_file)
        return

    def svm_predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size):

        features = self.extract_feature(src_img, scale, patch_size, seeds, batch_size)

        model = {5: "svm_densenet121_500_128_0.9185.model",
                 20: "svm_densenet121_2000_256_0.9607.model",
                 40: "svm_densenet121_4000_256_0.8817.model", }

        model_file = self._params.PROJECT_ROOT + "/models/pytorch/{}".format(model[int(scale)])
        classifier = joblib.load(model_file)
        predicted_tag = classifier.predict(features)

        return predicted_tag
