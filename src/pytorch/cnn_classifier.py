#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import datetime
import os
from abc import ABCMeta, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
# from torchsummary import summary
from modelsummary import summary
from sklearn import metrics
from torch.autograd import Variable
from torchvision import models

from core import Block
from core.util import latest_checkpoint
from core.util import read_csv_file,read_DSC_csv_file
from pytorch.image_dataset import Image_Dataset, DSC_Image_Dataset
from pytorch.loss_function import CenterLoss, LGMLoss, LGMLoss_v0
from pytorch.net import DenseNet, SEDenseNet, ExtendedDenseNet, DMC_DenseNet
from pytorch.net import Simple_CNN
from pytorch.util import get_image_blocks_itor, get_image_blocks_batch_normalize_itor, \
    get_image_file_batch_normalize_itor, get_image_blocks_dsc_itor


class BaseClassifier(object, metaclass=ABCMeta):
    def __init__(self, params, model_name, patch_type, **kwargs):
        '''
        初始化
        :param params: 系统参数
        :param model_name: 分类器算法的代号
        :param patch_type: 分类器处理的图块类型的代号
        '''
        self._params = params
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        # 在子类构造时初始化
        self.model_name = model_name
        self.num_classes = 0
        self.image_size = 0

        self.model_root = "{}/models/pytorch/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)

        self.use_GPU = True
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if self.use_GPU else "cpu")

        self.model = None
        if 'normalization' in kwargs:
            self.normal_func = kwargs["normalization"]
        else:
            self.normal_func = None

        if 'augmentation' in kwargs:
            self.augment_func = kwargs["augmentation"]
        else:
            self.augment_func = None

        if 'special_norm' in kwargs:
            self.special_norm_mode = kwargs["special_norm"]
        else:
            self.special_norm_mode = -1

    @abstractmethod
    def create_initial_model(self):
        '''
        生成初始化的模型
        :return:网络模型
        '''
        pass

    def load_model(self, model_file):
        '''
        加载模型
        :param model_file: 模型文件
        :return: 网络模型
        '''
        if model_file is None:
            checkpoint_dir = self.model_root
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            model_file = latest_checkpoint(checkpoint_dir)

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            load_object = torch.load(model_file)
            if isinstance(load_object, dict):
                model = self.create_initial_model()
                model.load_state_dict(torch.load(model_file))
            else:
                model = load_object
        else:
            model = self.create_initial_model()
        return model

    # 标准的训练过程
    def train_model(self, samples_name, class_weight, augment_func, batch_size, epochs):
        '''
        训练模型
        :param samples_name: 自制训练集的代号
        :param batch_size: 每批的图片数量
        :param epochs:epoch数量
        :return:
        '''
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        else:
            train_data, test_data = self.load_custom_data(samples_name, augment_func=augment_func)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        loss_func = nn.CrossEntropyLoss(weight=class_weight)

        if self.use_GPU:
            model.to(self.device)
            loss_func.to(self.device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, weight_decay = 0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            total_loss = 0
            starttime = datetime.datetime.now()
            for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

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

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)
                    print('%d / %d ==> Loss: %.4f | Acc: %.4f ,  remaining time: %d (s)'
                          % (step, train_data_len, running_loss, running_corrects.double()/b_x.size(0), remaining_time))

            scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            model.eval()
            # 开始评估
            for x, y in test_loader:
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)
                loss = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects += torch.sum(preds == b_y.data)

            test_data_len = test_data.__len__()
            epoch_loss=running_loss / test_data_len
            epoch_acc=running_corrects.double() / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc),
                       )
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

    def load_custom_data(self, samples_name, augment_func = None):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        patch_root = self._params.PATCHS_ROOT_PATH[samples_name[0]]
        sample_filename = samples_name[1]
        train_list = "{}/{}_train.txt".format(patch_root, sample_filename)
        test_list = "{}/{}_test.txt".format(patch_root, sample_filename)

        Xtrain, Ytrain = read_csv_file(patch_root, train_list)
        # Xtrain, Ytrain = Xtrain[:400], Ytrain[:400] # for debug
        train_data = Image_Dataset(Xtrain, Ytrain,transform = None, augm = augment_func, norm = None)

        Xtest, Ytest = read_csv_file(patch_root, test_list)
        # Xtest, Ytest = Xtest[:60], Ytest[:60]  # for debug
        test_data = Image_Dataset(Xtest, Ytest)
        return  train_data, test_data

    def loading_test_dataset(self, samples_name, batch_size, max_count, special_norm_mode = -1):
        test_list = "{}/{}".format(self._params.PATCHS_ROOT_PATH[samples_name[0]], samples_name[1])
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH[samples_name[0]], test_list)

        if max_count is not None:
            Xtest, Ytest = Xtest[:max_count], Ytest[:max_count]  # for debug

        if special_norm_mode == 0:
            # 自定义的数据加载方式
            test_loader = get_image_file_batch_normalize_itor(Xtest, Ytest, batch_size, self.normal_func, False)
        elif special_norm_mode == 1:
            test_loader = get_image_file_batch_normalize_itor(Xtest, Ytest, batch_size, self.normal_func, True)
        else:
            test_data = Image_Dataset(Xtest, Ytest, norm = self.normal_func)
            test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
                                          shuffle=False, num_workers=self.NUM_WORKERS)

        return test_loader, Xtest, Ytest

    def evaluate_model(self, samples_name, model_file, batch_size, max_count):
        test_loader, Xtest, Ytest  = self.loading_test_dataset(samples_name, batch_size, max_count, self.special_norm_mode)

        model = self.load_model(model_file=model_file)
        # 关闭求导，节约大量的显存
        for param in model.parameters():
            param.requires_grad = False
        print(model)

        model.to(self.device)
        model.eval()

        predicted_tags = []
        features = []
        len_y = len(Ytest)
        if len_y % batch_size == 0:
            test_data_len = len_y // batch_size
        else:
            test_data_len = len_y // batch_size + 1

        starttime = datetime.datetime.now()
        for step, (x, _) in enumerate(test_loader):
            b_x = Variable(x.to(self.device))  # batch x

            output = model(b_x)  # model最后不包括一个softmax层
            output_softmax = nn.functional.softmax(output, dim=1)
            probs, preds = torch.max(output_softmax, 1)

            predicted_tags.extend(preds.cpu().numpy())
            features.extend(output.cpu().numpy())

            endtime = datetime.datetime.now()
            remaining_time = (test_data_len - step) * (endtime - starttime).seconds / (step + 1)
            print('predicting => %d / %d , remaining time: %d (s)' % (step + 1, test_data_len, remaining_time))

        Ytest = np.array(Ytest)
        predicted_tags = np.array(predicted_tags)
        print("%s Classification report for classifier :\n%s\n"
              % (self.model_name, metrics.classification_report(Ytest, predicted_tags, digits=4)))

        return Xtest, Ytest, predicted_tags, features

    def evaluate_models(self, samples_name, model_directory, batch_size, max_count):
        test_loader, Xtest, Ytest = self.loading_test_dataset(samples_name, batch_size, max_count,
                                                              self.special_norm_mode)

        len_y = len(Ytest)
        if len_y % batch_size == 0:
            test_data_len = len_y // batch_size
        else:
            test_data_len = len_y // batch_size + 1

        filename = []
        loss = []
        train_accuracy = []
        test_accuracy = []

        for model_file in os.listdir(model_directory):
            file_name = os.path.splitext(model_file)[0]
            ext_name = os.path.splitext(model_file)[1]
            if ext_name == ".pth":
                value = file_name.split("-")
                if len(value) >= 4:
                    filename.append(file_name)
                    loss.append(float(value[2]))
                    train_accuracy.append(float(value[3]))

                model = self.load_model(model_file="{}/{}".format(model_directory, model_file))
                # 关闭求导，节约大量的显存
                for param in model.parameters():
                    param.requires_grad = False

                model.to(self.device)
                model.eval()
                predicted_tags = []
                starttime = datetime.datetime.now()
                for step, (x, _) in enumerate(test_loader):
                    b_x = Variable(x.to(self.device))  # batch x

                    output = model(b_x)  # model最后不包括一个softmax层
                    output_softmax = nn.functional.softmax(output, dim=1)
                    probs, preds = torch.max(output_softmax, 1)

                    predicted_tags.extend(preds.cpu().numpy())

                    endtime = datetime.datetime.now()
                    remaining_time = (test_data_len - step) * (endtime - starttime).seconds / (step + 1)
                    print('predicting => %d / %d , remaining time: %d (s)' % (step + 1, test_data_len, remaining_time))

                test_accu = np.sum(np.array(predicted_tags) == np.array(Ytest), dtype=np.float)/len_y
                # print(predicted_tags)
                # print(Ytest)
                print(file_name, test_accu)
                test_accuracy.append(test_accu)


        data = {'filename': filename, 'loss': loss, 'train accuracy': train_accuracy,
                'test accuracy': test_accuracy}
        df = pd.DataFrame(data, columns=['filename', 'loss', 'train accuracy', 'test accuracy'])
        result = df.sort_values(['loss', 'train accuracy','test accuracy'], ascending=[True, False, False])

        return result

    def evaluate_accuracy_based_slice(self, Xtest, results, y_true_set):
        slice_result = {}
        b = Block()
        for file_name, y, true_y in zip(Xtest, results, y_true_set):
            b.decoding(file_name, 256, 256)
            if b.slice_number in slice_result.keys():
                slice_result[b.slice_number].append(y == true_y)
            else:
                slice_result[b.slice_number] = [y == true_y]

        for slice_name in sorted(slice_result.keys()):
            value = slice_result[slice_name]
            count = len(value)
            accu = float(sum(value)) / count
            print("{} => accu ={:.4f}, count = {}".format(slice_name, accu, count))
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
        if self.special_norm_mode == 0:
            seeds_itor = get_image_blocks_batch_normalize_itor(src_img, scale, seeds, patch_size, patch_size,
                                                               batch_size,
                                                               normalization=self.normal_func, dynamic_update=False)
        elif self.special_norm_mode == 1:
            seeds_itor = get_image_blocks_batch_normalize_itor(src_img, scale, seeds, patch_size, patch_size,
                                                               batch_size,
                                                               normalization=self.normal_func, dynamic_update=True)
        else:
            seeds_itor = get_image_blocks_itor(src_img, scale, seeds, patch_size, patch_size, batch_size,
                                               normalization=self.normal_func)


        if self.model is None:
            self.model = self.load_pretrained_model_on_predict()
            self.model.to(self.device)
            self.model.eval()

        len_seeds = len(seeds)
        data_len = len(seeds) // batch_size
        if len_seeds % batch_size > 0:
            data_len += 1

        probability = []
        prediction = []
        low_dim_features = []
        for step, x in enumerate(seeds_itor):
            b_x = Variable(x.to(self.device))

            output = self.model(b_x) # model最后不包括一个softmax层
            output_softmax = nn.functional.softmax(output, dim =1)
            probs, preds = torch.max(output_softmax, 1)

            low_dim_features.extend(output.cpu().numpy())
            probability.extend(probs.cpu().numpy())
            prediction.extend(preds.cpu().numpy())
            print('predicting => %d / %d ' % (step + 1, data_len))

        return probability, prediction, low_dim_features

    @abstractmethod
    def load_pretrained_model_on_predict(self):
        pass

    def export_ONNX_model(self):
        '''
        :return:
        '''

        # import torch.onnx
        batch_size = 1  # just a random number

        # Input to the model
        x = Variable(torch.randn(batch_size, 3, 256, 256), requires_grad=True)

        torch_model = self.create_initial_model()
        # torch_model = torch.load(model_file, map_location=lambda storage, loc: storage)
        # torch_model.MultiTask = True

        torch_model.eval()
        # Export the model
        torch_out = torch.onnx.export(torch_model,  # model being run
                                       x,  # model input (or a tuple for multiple inputs)
                                       "{}.onnx".format(self.model_name),  # where to save the model (can be a file or file-like object)
                                      export_params=False, verbose=False)  # store the trained parameter weights inside the model file

    def export_tensorboard_model(self):
        torch_model = self.create_initial_model()
        batch_size = 1
        x = Variable(torch.randn(batch_size, 3, 256, 256), requires_grad=True)
        torch_model.eval()

        from tensorboardX import SummaryWriter
        with SummaryWriter(comment="{}".format(self.model_name)) as w:
            w.add_graph(torch_model, (x, ))

    # def visualize_features(self, features, true_labels, predicted_tags):
    #     features = np.array(features)
    #
    #     c = ['#00ff00', '#ff0000', '#ffff00', '#00ffff', '#0000ff',
    #          '#99ff00', '#ff0099', '#999900', '#009900', '#009999']
    #     plt.clf()
    #     for i in range(2):
    #         # feat = features[np.logical_and(true_labels == i, true_labels == predicted_tags)]
    #         # plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i])
    #         #
    #         # feat = features[np.logical_and(true_labels == i, true_labels != predicted_tags)]
    #         # plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i + 5])
    #         feat = features[true_labels == i]
    #         center = np.mean(feat, axis=0)
    #         plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i], alpha=0.6)
    #         print("the center of Label ", i, " = ",center)
    #         plt.plot(center[0], center[1], '*', c=c[i + 5], markersize=16)
    #
    #     # plt.legend(['true_cancer', 'false_cancer', 'true_normal', 'false_normal'], loc='upper right')
    #     plt.legend(['normal', 'center of normal ', 'cancer', 'center of cancer ',], loc='upper right')
    #     plt.show()

    def visualize_features(self, Xtest, features, true_labels):
        slice_cancer_result = {}
        slice_normal_result = {}

        b = Block()
        for file_name, f, true_y in zip(Xtest, features, true_labels):
            b.decoding(file_name, 256, 256)
            if true_y == 0:
                if b.slice_number in slice_normal_result.keys():
                    slice_normal_result[b.slice_number].append(f)
                else:
                    slice_normal_result[b.slice_number] = [f]
            else:
                if b.slice_number in slice_cancer_result.keys():
                    slice_cancer_result[b.slice_number].append(f)
                else:
                    slice_cancer_result[b.slice_number] = [f]

        count = len(slice_cancer_result) + len(slice_normal_result)

        cmap = matplotlib.cm.get_cmap('Spectral')
        k = np.linspace(0,1.0, count)
        random_color = cmap(k)

        plt.clf()
        n = 0
        label = ["c", "n"]
        legends = []
        for state, result in enumerate([slice_cancer_result, slice_normal_result]):
            for id, f in result.items():
                f = np.array(f)
                color = random_color[n]
                plt.plot(f[:, 0], f[:, 1], '.', c=color, alpha=0.6)
                n += 1
                legends.append("{}.{}".format(label[state], id))

        plt.legend(legends, loc='upper right')
        plt.show()


# 输入为RGB三通道图像，单输出的分类器
######################################################################################################################
############       single task            #########
######################################################################################################################
class Simple_Classifier(BaseClassifier):
    def __init__(self, params, model_name, patch_type, **kwargs):
        super(Simple_Classifier, self).__init__(params, model_name, patch_type,**kwargs)

        if self.patch_type in ["1000_256", "2000_256", "4000_256", "x_256", "msc_256"]:
            self.num_classes = 2
            self.image_size = 256
        elif self.patch_type == "cifar10":
            self.num_classes = 10
            self.image_size = 32
        elif self.patch_type == "cifar100":
            self.num_classes = 100
            self.image_size = 32

    def create_initial_model(self):
        def create_densenet(depth, gvp_out_size):
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
                    small_inputs=True,  # 32 x 32的图片为True
                    gvp_out_size=gvp_out_size,
                    efficient=True,
                )
            elif self.patch_type in ["500_128", "2000_256", "4000_256", "x_256"]:  # 256 x 256
                # Models
                model = DenseNet(
                    growth_rate=12,
                    block_config=block_config,
                    num_classes=self.num_classes,
                    small_inputs=False,  # 32 x 32的图片为True
                    gvp_out_size=gvp_out_size,
                    efficient=False,
                )
            return model

        def create_e_densenet(depth, growth_rate, gvp_out_size):
            # Get densenet configuration
            if (depth - 4) % 3:
                raise Exception('Invalid depth')
            block_config = [(depth - 4) // 6 for _ in range(3)]

            # Models
            model = ExtendedDenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_classes=self.num_classes,
                # drop_rate=0.2,
                gvp_out_size=gvp_out_size,
            )
            return model

        def create_se_densenet(depth, gvp_out_size):
            # Get densenet configuration
            if (depth - 4) % 3:
                raise Exception('Invalid depth')
            block_config = [(depth - 4) // 6 for _ in range(3)]

            # Models
            model = SEDenseNet(
                growth_rate=12,
                block_config=block_config,
                num_classes=self.num_classes,
                gvp_out_size=gvp_out_size,
            )
            return model

        if self.model_name == "simple_cnn":
            model = Simple_CNN(self.num_classes, self.image_size)
        elif self.model_name == "densenet_22":
            model = create_densenet(depth=22, gvp_out_size=1)
        elif self.model_name == "densenet_40":
            # model = create_densenet(depth=40, gvp_out_size=(2,2))
            model = create_densenet(depth=40, gvp_out_size=1)
        elif self.model_name == "se_densenet_22":
            model = create_se_densenet(depth=22, gvp_out_size=1)
        elif self.model_name =="se_densenet_40":
            model= create_se_densenet(depth=40, gvp_out_size=(2,2))
        elif self.model_name =="resnet_18":
            model = models.resnet18(pretrained=False, num_classes=2)
        elif self.model_name =="resnet_34":
            model = models.resnet34(pretrained=False, num_classes=2)
        elif self.model_name == "e_densenet_22":
            model = create_e_densenet(depth=22, growth_rate=12, gvp_out_size=1)
        elif self.model_name == "e_densenet_40":
            model = create_e_densenet(depth=40, growth_rate=16, gvp_out_size=1)
        return model

    def load_pretrained_model_on_predict(self):
        '''
        加载已经训练好的存盘网络文件
        :param patch_type: 分类器处理图块的类型
        :return: 网络模型
        '''
        net_file = {
            "simple_cnn_4000_256": "simple_cnn_cps-0010-0.1799-0.9308.pth",
            "densenet_22_4000_256": "densenet_22_4000_256_cp-0005-0.1423-0.9486.pth",
            "se_densenet_22_4000_256":"se_densenet_22_cp-0001-0.1922-0.9223-0.9094.pth",
            "se_densenet_40_4000_256":"se_densenet_40_4000_256_cp-0002-0.1575-0.9436.pth",
            "e_densenet_22_4000_256":"e_densenet_22_4000_256_cp-0002-0.0996-0.9634.pth",
            "e_densenet_40_2000_256":"e_densenet_40_2000_256_cp-0009-0.1141-0.9594.pth"
        }

        model_code = "{}_{}".format(self.model_name, self.patch_type)
        model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[model_code])
        model = self.load_model(model_file=model_file)

        # 关闭求导，节约大量的显存
        for param in model.parameters():
            param.requires_grad = False
        return model

    def train_model_A1(self, samples_name, class_weight, augment_func, batch_size, loss_weight, epochs):
        '''
        训练模型
        :param samples_name: 自制训练集的代号
        :param batch_size: 每批的图片数量
        :param epochs:epoch数量
        :return:
        '''
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        else:
            train_data, test_data = self.load_custom_data(samples_name, augment_func=augment_func)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        classifi_loss = nn.CrossEntropyLoss(weight=class_weight)
        center_loss = CenterLoss(self.num_classes, 2)
        if self.use_GPU:
            model.to(self.device)
            classifi_loss.to(self.device)
            center_loss.to(self.device)

        # optimzer4nn
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        classifi_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifi_optimizer, mode='min',
                                                               factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
        # optimzer4center
        optimzer4center = torch.optim.SGD(center_loss.parameters(), lr=0.1)

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            total_loss = 0

            starttime = datetime.datetime.now()
            for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)  # cnn output is features, not logits
                # cross entropy loss + center loss
                loss = classifi_loss(output, b_y) + loss_weight * center_loss(b_y, output)

                classifi_optimizer.zero_grad()  # clear gradients for this training step
                optimzer4center.zero_grad()
                loss.backward()  # backpropagation, compute gradients
                classifi_optimizer.step()
                optimzer4center.step()

                # 数据统计
                _, preds = torch.max(output, 1)

                running_loss = loss.item()
                running_corrects = torch.sum(preds == b_y.data)
                total_loss += running_loss

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)
                    print('%d / %d ==> Loss: %.4f | Acc: %.4f ,  remaining time: %d (s)'
                          % (step, train_data_len, running_loss, running_corrects.double()/b_x.size(0), remaining_time))

            scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            model.eval()
            # 开始评估
            for x, y in test_loader:
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)
                loss = classifi_loss(output, b_y)

                _, preds = torch.max(output, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects += torch.sum(preds == b_y.data)

            test_data_len = test_data.__len__()
            epoch_loss=running_loss / test_data_len
            epoch_acc=running_corrects.double() / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc),
                       )
        return

    def train_model_A2(self, samples_name, class_weight, augment_func, batch_size, loss_weight, epochs):
        '''
        训练模型
        :param samples_name: 自制训练集的代号
        :param batch_size: 每批的图片数量
        :param epochs:epoch数量
        :return:
        '''
        if self.patch_type in ["cifar10", "cifar100"]:
            train_data, test_data = self.load_cifar_data(self.patch_type)
        else:
            train_data, test_data = self.load_custom_data(samples_name, augment_func=augment_func)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        classifi_loss = nn.CrossEntropyLoss(weight=class_weight)
        lgm_loss = LGMLoss(self.num_classes, 2, 1.00)
        if self.use_GPU:
            model.to(self.device)
            classifi_loss.to(self.device)
            lgm_loss.to(self.device)

        # optimzer4nn
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # classifi_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifi_optimizer, mode='min',
                                                               factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
        # optimzer4center
        optimzer4center = torch.optim.SGD(lgm_loss.parameters(), lr=0.1, momentum=0.9)

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            total_loss = 0

            starttime = datetime.datetime.now()
            for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)  # cnn output is features, not logits
                # cross entropy loss + center loss
                # loss = classifi_loss(output, b_y) + loss_weight * center_loss(b_y, output)
                logits, mlogits, likelihood = lgm_loss(output, b_y)
                loss = classifi_loss(mlogits, b_y) + loss_weight * likelihood

                classifi_optimizer.zero_grad()  # clear gradients for this training step
                optimzer4center.zero_grad()
                loss.backward()  # backpropagation, compute gradients
                classifi_optimizer.step()
                optimzer4center.step()

                # 数据统计
                _, preds = torch.max(output, 1)

                running_loss = loss.item()
                running_corrects = torch.sum(preds == b_y.data)
                total_loss += running_loss

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)

                    tmp = lgm_loss.log_covs
                    norm = torch.sum(torch.mul(tmp, tmp))
                    norm_value = norm.item()

                    print('%d / %d ==> Cov_norm %.4f | Loss: %.4f | Acc: %.4f ,  remaining time: %d (s)'
                          % (step, train_data_len, norm_value, running_loss, running_corrects.double()/b_x.size(0),
                             remaining_time))

            scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            model.eval()
            # 开始评估
            for x, y in test_loader:
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)
                loss = classifi_loss(output, b_y)

                _, preds = torch.max(output, 1)
                running_loss += loss.item() * b_x.size(0)
                running_corrects += torch.sum(preds == b_y.data)

            test_data_len = test_data.__len__()
            epoch_loss=running_loss / test_data_len
            epoch_acc=running_corrects.double() / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc),
                       )
        return


class DMC_Classifier(BaseClassifier):
    def __init__(self, params, model_name, patch_type, **kwargs):
        super(DMC_Classifier, self).__init__(params, model_name, patch_type, **kwargs)

        if self.patch_type in ["2040_256"]:
            self.num_classes = 2
            self.image_size = 256

    def create_initial_model(self):
        def create_dsc_densenet(depth, growth_rate, gvp_out_size):
            # Get densenet configuration
            if (depth - 4) % 3:
                raise Exception('Invalid depth')
            block_config = [(depth - 4) // 6 for _ in range(3)]

            # Models
            model = DMC_DenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_classes=self.num_classes,
                # drop_rate=0.2,
                gvp_out_size=gvp_out_size,
                efficient=False,
            )
            return model

        if self.model_name == "dsc_densenet_40":
            model = create_dsc_densenet(depth=40, growth_rate=16, gvp_out_size=1)

        return model

    def load_pretrained_model_on_predict(self):
        '''
        加载已经训练好的存盘网络文件
        :param patch_type: 分类器处理图块的类型
        :return: 网络模型
        '''
        net_file = {
            "dsc_densenet_40_2040_256":"dsc_densenet_40_2040_256_cp-0016-0.0738-0.9724-0.9702-0.9521.pth"
        }

        model_code = "{}_{}".format(self.model_name, self.patch_type)
        model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[model_code])
        model = self.load_model(model_file=model_file)

        # 关闭求导，节约大量的显存
        for param in model.parameters():
            param.requires_grad = False
        return model

    def load_custom_data(self, samples_name, augment_func = None):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :return:用于train和test的两个Sequence
        '''
        patch_root = self._params.PATCHS_ROOT_PATH[samples_name[0]]
        sample_filename = samples_name[1]
        train_list = "{}/{}_train.txt".format(patch_root, sample_filename)
        test_list = "{}/{}_test.txt".format(patch_root, sample_filename)

        Xtrain, Ytrain = read_DSC_csv_file(patch_root, train_list)
        # Xtrain, Ytrain = Xtrain[:40], Ytrain[:40] # for debug
        train_data = DSC_Image_Dataset(Xtrain, Ytrain, transform=None)

        Xtest, Ytest = read_DSC_csv_file(patch_root, test_list)
        # Xtest, Ytest = Xtest[:60], Ytest[:60]  # for debug
        test_data = DSC_Image_Dataset(Xtest, Ytest, transform=None)
        return train_data, test_data

    def train_model(self, samples_name, class_weight, batch_size, epochs):
        '''
        训练模型
        :param samples_name: 自制训练集的代号
        :param batch_size: 每批的图片数量
        :param epochs:epoch数量
        :return:
        '''

        train_data, test_data = self.load_custom_data(samples_name,)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)),
                torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True, show_hierarchical=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        loss_func = nn.CrossEntropyLoss(weight=class_weight)

        if self.use_GPU:
            model.to(self.device)
            loss_func.to(self.device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        optimizer_x2040 = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_xDS = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, weight_decay = 0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                        factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            starttime = datetime.datetime.now()
            for step, ((x20, x40), (y20, y40, y)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)  # cnn output
                loss1 = loss_func(output, b_y)
                loss2 = 0.5 * (loss_func(output20, b_y20) + loss_func(output40, b_y40)) # cross entropy loss
                optimizer_x2040.zero_grad()  # clear gradients for this training step
                optimizer_xDS.zero_grad()

                loss1.backward(retain_graph=True)  # backpropagation, compute gradients
                loss2.backward()
                optimizer_x2040.step()
                optimizer_xDS.step()

                # 数据统计
                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)

                running_loss = loss1.item()
                running_corrects = torch.sum(preds == b_y.data)

                running_loss2 = loss2.item()
                running_corrects2 = torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 = torch.sum(preds_x40 == b_y40.data).item()

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)
                    print('%d / %d ==> Loss1: %.4f | Acc: %.4f , Loss2: %.4f | Acc_x20: %.4f | Acc_x40: %.4f, remaining time: %d (s)'
                          % (step, train_data_len,
                             running_loss, float(running_corrects) / b_y.size(0),
                             running_loss2, float(running_corrects2) / b_y.size(0), float(running_corrects4) / b_y.size(0),
                             remaining_time))

            # scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            running_corrects2=0
            running_corrects4=0
            model.eval()
            # 开始评估
            for (x20, x40), (y20, y40, y) in test_loader:
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)
                loss1 = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)
                running_loss += loss1.item() * b_y.size(0)
                running_corrects += torch.sum(preds == b_y.data).item()
                running_corrects2 += torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 += torch.sum(preds_x40 == b_y40.data).item()

            test_data_len = test_data.__len__()
            epoch_loss = running_loss / test_data_len
            epoch_acc = float(running_corrects) / test_data_len
            epoch_acc20 = float(running_corrects2) / test_data_len
            epoch_acc40 = float(running_corrects4) / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc, epoch_acc20, epoch_acc40),
                       )
        return

    def train_model_A2(self, samples_name, class_weight, batch_size, loss_weight, epochs):
        train_data, test_data = self.load_custom_data(samples_name,)

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)),
                torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True, show_hierarchical=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        loss_func = nn.CrossEntropyLoss(weight=class_weight)
        lgm_loss = LGMLoss(self.num_classes, 2, 1.00)
        # lgm_loss = LGMLoss_v0(self.num_classes, 2, 1.00)

        if self.use_GPU:
            model.to(self.device)
            loss_func.to(self.device)
            lgm_loss.to(self.device)

        optimizer_x2040 = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_xDS = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer_x2040 = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)
        # optimizer_xDS = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, weight_decay = 0.001)

        # optimzer4center
        optimzer4center = torch.optim.SGD(lgm_loss.parameters(), lr=0.1, momentum=0.9)

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            starttime = datetime.datetime.now()
            for step, ((x20, x40), (y20, y40, y)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)  # cnn output
                
                output2040 = torch.cat((output20, output40),0)
                b_y2040 = torch.cat((b_y20, b_y40), 0)
                # logits20, mlogits20, likelihood20 = lgm_loss(output20, b_y20)
                # logits40, mlogits40, likelihood40 = lgm_loss(output40, b_y40)
                logits2040, mlogits2040, likelihood2040 = lgm_loss(output2040, b_y2040)

                # 组合特征的分类器的loss
                loss1 = loss_func(output, b_y)

                # 单个倍镜下的分类器的Loss
                m = 0.5
                # cross entropy loss
                # loss2 = m * loss_func(output20, b_y20) + (1 - m) * loss_func(output40, b_y40) + loss_weight * (likelihood20 + likelihood40)
                loss2 = m * loss_func(output20, b_y20) + (1 - m) * loss_func(output40, b_y40) + loss_weight * likelihood2040

                optimizer_x2040.zero_grad()  # clear gradients for this training step
                optimizer_xDS.zero_grad()
                optimzer4center.zero_grad()

                loss1.backward(retain_graph=True)  # backpropagation, compute gradients,
                loss2.backward()

                optimizer_x2040.step()
                optimizer_xDS.step()
                optimzer4center.step()

                # 数据统计
                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)

                running_loss = loss1.item()
                running_corrects = torch.sum(preds == b_y.data)

                running_loss2 = loss2.item()
                running_corrects2 = torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 = torch.sum(preds_x40 == b_y40.data).item()

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)

                    tmp = lgm_loss.log_covs
                    norm = torch.sum(torch.mul(tmp, tmp))
                    norm_value = norm.item()

                    print('%d / %d ==>  Cov_norm %.4f | Loss1: %.4f | Acc: %.4f , Loss2: %.4f | Acc_x20: %.4f | Acc_x40: %.4f, remaining time: %d (s)'
                          % (step, train_data_len, norm_value,
                             running_loss, float(running_corrects) / b_y.size(0),
                             running_loss2, float(running_corrects2) / b_y.size(0), float(running_corrects4) / b_y.size(0),
                             remaining_time))

            running_loss=0.0
            running_corrects=0
            running_corrects2=0
            running_corrects4=0
            model.eval()
            # 开始评估
            for (x20, x40), (y20, y40, y) in test_loader:
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)
                loss1 = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)
                running_loss += loss1.item() * b_y.size(0)
                running_corrects += torch.sum(preds == b_y.data).item()
                running_corrects2 += torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 += torch.sum(preds_x40 == b_y40.data).item()

            test_data_len = test_data.__len__()
            epoch_loss = running_loss / test_data_len
            epoch_acc = float(running_corrects) / test_data_len
            epoch_acc20 = float(running_corrects2) / test_data_len
            epoch_acc40 = float(running_corrects4) / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc, epoch_acc20, epoch_acc40),
                       )
        return

    def train_model_A3(self, samples_name, class_weight, batch_size, loss_weight, epochs):

        train_data, test_data = self.load_custom_data(samples_name, )

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        # print(model)
        summary(model, torch.zeros((1, 3, self.image_size, self.image_size)),
                torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True, show_hierarchical=True)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)
        loss_func = nn.CrossEntropyLoss(weight=class_weight)
        # lgm_loss = LGMLoss(self.num_classes, 2, 1.00)
        lgm_loss = LGMLoss_v0(self.num_classes, 2, 1.00)

        if self.use_GPU:
            model.to(self.device)
            loss_func.to(self.device)
            lgm_loss.to(self.device)

        optimizer_x2040 = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_xDS = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer_x2040 = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)
        # optimizer_xDS = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, weight_decay = 0.001)

        # optimzer4center
        optimzer4center = torch.optim.SGD(lgm_loss.parameters(), lr=0.1, momentum=0.9)

        # training and testing
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            # 开始训练
            train_data_len = len(train_loader)
            starttime = datetime.datetime.now()
            for step, ((x20, x40), (y20, y40, y)) in enumerate(
                    train_loader):  # 分配 batch data, normalize x when iterate train_loader
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)  # cnn output

                output2040 = torch.cat((output20, output40), 0)
                b_y2040 = torch.cat((b_y20, b_y40), 0)
                # logits20, mlogits20, likelihood20 = lgm_loss(output20, b_y20)
                # logits40, mlogits40, likelihood40 = lgm_loss(output40, b_y40)
                logits2040, mlogits2040, likelihood2040 = lgm_loss(output2040, b_y2040)

                # 组合特征的分类器的loss
                loss1 = loss_func(output, b_y)

                # 单个倍镜下的分类器的Loss
                m = 0.5
                # cross entropy loss
                # loss2 = m * loss_func(output20, b_y20) + (1 - m) * loss_func(output40, b_y40) + loss_weight * (likelihood20 + likelihood40)
                loss2 = m * loss_func(output20, b_y20) + (1 - m) * loss_func(output40,
                                                                             b_y40) + loss_weight * likelihood2040

                optimizer_x2040.zero_grad()  # clear gradients for this training step
                optimizer_xDS.zero_grad()
                optimzer4center.zero_grad()

                loss1.backward(retain_graph=True)  # backpropagation, compute gradients,
                loss2.backward()

                optimizer_x2040.step()
                optimizer_xDS.step()
                optimzer4center.step()

                # 数据统计
                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)

                running_loss = loss1.item()
                running_corrects = torch.sum(preds == b_y.data)

                running_loss2 = loss2.item()
                running_corrects2 = torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 = torch.sum(preds_x40 == b_y40.data).item()

                if step % 50 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step) * (endtime - starttime).seconds / (step + 1)

                    # tmp = lgm_loss.log_covs
                    # norm = torch.sum(torch.mul(tmp, tmp))
                    # norm_value = norm.item()

                    print(
                        '%d / %d ==> Loss1: %.4f | Acc: %.4f , Loss2: %.4f | Acc_x20: %.4f | Acc_x40: %.4f, remaining time: %d (s)'
                        % (step, train_data_len, running_loss, float(running_corrects) / b_y.size(0),
                           running_loss2, float(running_corrects2) / b_y.size(0),
                           float(running_corrects4) / b_y.size(0),
                           remaining_time))

                    class_centers = lgm_loss.centers.detach().cpu().numpy()
                    print("class_centers", class_centers)

            running_loss = 0.0
            running_corrects = 0
            running_corrects2 = 0
            running_corrects4 = 0
            model.eval()
            # 开始评估
            for (x20, x40), (y20, y40, y) in test_loader:
                b_x20 = Variable(x20.to(self.device))
                b_x40 = Variable(x40.to(self.device))
                b_y20 = Variable(y20.to(self.device))
                b_y40 = Variable(y40.to(self.device))
                b_y = Variable(y.to(self.device))

                output20, output40, output = model(b_x20, b_x40)
                loss1 = loss_func(output, b_y)

                _, preds = torch.max(output, 1)
                _, preds_x20 = torch.max(output20, 1)
                _, preds_x40 = torch.max(output40, 1)
                running_loss += loss1.item() * b_y.size(0)
                running_corrects += torch.sum(preds == b_y.data).item()
                running_corrects2 += torch.sum(preds_x20 == b_y20.data).item()
                running_corrects4 += torch.sum(preds_x40 == b_y40.data).item()

            test_data_len = test_data.__len__()
            epoch_loss = running_loss / test_data_len
            epoch_acc = float(running_corrects) / test_data_len
            epoch_acc20 = float(running_corrects2) / test_data_len
            epoch_acc40 = float(running_corrects4) / test_data_len

            torch.save(model.state_dict(), self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type, epoch + 1, epoch_loss, epoch_acc, epoch_acc20, epoch_acc40),
                       )
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
        seeds_itor = get_image_blocks_dsc_itor(src_img, scale, seeds, patch_size, patch_size, batch_size,)


        if self.model is None:
            self.model = self.load_pretrained_model_on_predict()
            self.model.to(self.device)
            self.model.eval()

        len_seeds = len(seeds)
        data_len = len(seeds) // batch_size
        if len_seeds % batch_size > 0:
            data_len += 1

        probability = []
        prediction = []
        low_dim_features = []
        for step, (x20, x40) in enumerate(seeds_itor):
            b_x20 = Variable(x20.to(self.device))
            b_x40 = Variable(x40.to(self.device))

            out20, out40, output = self.model(b_x20, b_x40) # model最后不包括一个softmax层
            output_softmax = nn.functional.softmax(output, dim =1)
            probs, preds = torch.max(output_softmax, 1)

            low_dim_features.extend(output.cpu().numpy())
            probability.extend(probs.cpu().numpy())
            prediction.extend(preds.cpu().numpy())
            print('predicting => %d / %d ' % (step + 1, data_len))

        return probability, prediction, low_dim_features

    def evaluate_model(self, samples_name, model_file, batch_size):
        train_data, test_data = self.load_custom_data(samples_name, )

        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        if self.model is None:
            self.model = self.load_pretrained_model_on_predict()
            self.model.to(self.device)
            self.model.eval()

        running_corrects = 0
        running_corrects2 = 0
        running_corrects4 = 0

        starttime = datetime.datetime.now()
        for (x20, x40), (y20, y40, y) in test_loader:
            b_x20 = Variable(x20.to(self.device))
            b_x40 = Variable(x40.to(self.device))
            b_y20 = Variable(y20.to(self.device))
            b_y40 = Variable(y40.to(self.device))
            b_y = Variable(y.to(self.device))

            output20, output40, output = self.model(b_x20, b_x40)

            _, preds = torch.max(output, 1)
            _, preds_x20 = torch.max(output20, 1)
            _, preds_x40 = torch.max(output40, 1)

            running_corrects += torch.sum(preds == b_y.data).item()
            running_corrects2 += torch.sum(preds_x20 == b_y20.data).item()
            running_corrects4 += torch.sum(preds_x40 == b_y40.data).item()

        endtime = datetime.datetime.now()

        test_data_len = test_data.__len__()
        epoch_acc = float(running_corrects) / test_data_len
        epoch_acc20 = float(running_corrects2) / test_data_len
        epoch_acc40 = float(running_corrects4) / test_data_len
        speed =  float(test_data_len) / (endtime - starttime).seconds

        print("acc, acc20, acc40, speed:", epoch_acc, epoch_acc20, epoch_acc40, speed)

    def calc_centers(self, samples_name, model_file, batch_size):
        train_data, test_data = self.load_custom_data(samples_name, )

        test_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        # test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
        #                               num_workers=self.NUM_WORKERS)

        if self.model is None:
            self.model = self.load_pretrained_model_on_predict()
            self.model.to(self.device)
            self.model.eval()

        features_20T = []
        features_20N = []
        features_40T = []
        features_40N = []

        for (x20, x40), (y20, y40, y) in test_loader:
            b_x20 = Variable(x20.to(self.device))
            b_x40 = Variable(x40.to(self.device))
            b_y20 = Variable(y20.to(self.device))
            b_y40 = Variable(y40.to(self.device))
            b_y = Variable(y.to(self.device))

            output20, output40, output = self.model(b_x20, b_x40)

            for f20, f40, yy in zip(output20.cpu().numpy(), output40.cpu().numpy(), y):
                if yy == 0:
                    features_20N.append(f20)
                    features_40N.append(f40)
                else:
                    features_20T.append(f20)
                    features_40T.append(f40)

        center_20N = np.mean(np.array(features_20N), axis=0)
        center_20T = np.mean(np.array(features_20T), axis=0)
        print("center_20 Normal", center_20N)
        print("center_20 Tumor", center_20T)

        center_40N = np.mean(np.array(features_40N), axis=0)
        center_40T = np.mean(np.array(features_40T), axis=0)
        print("center_40 Normal", center_40N)
        print("center_40 Tumor", center_40T)

# ######################################################################################################################
# ############       multi task            #########
# ######################################################################################################################
# class MultiTask_Classifier(BaseClassifier):
#     def __init__(self, params, model_name, patch_type, **kwargs):
#         super(MultiTask_Classifier, self).__init__(params, model_name, patch_type, **kwargs)
#
#         self.num_classes = (2, 3)
#         self.image_size = 256
#
#     def create_initial_model(self):
#         def create_se_densenet(self, depth):
#             # Get densenet configuration
#             if (depth - 4) % 3:
#                 raise Exception('Invalid depth')
#             block_config = [(depth - 4) // 6 for _ in range(3)]
#
#             # Models
#             model = SEDenseNet(
#                 growth_rate=12,
#                 block_config=block_config,
#                 num_classes=self.num_classes,
#                 gvp_out_size=1,
#             )
#             return model
#
#         if self.model_name == "se_densenet_22":
#             model = create_se_densenet(depth=22)
#         elif self.model_name == "se_densenet_40":
#             model = create_se_densenet(depth=40)
#
#         return model
#
#     def train_model(self, samples_name, augment_func, batch_size, epochs):
#         train_data, test_data = self.load_custom_data(samples_name)
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size,
#                                        shuffle=True, num_workers=self.NUM_WORKERS)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         summary(model, torch.zeros((1, 3, self.image_size, self.image_size)), show_input=True)
#
#         model.to(self.device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
#         # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
#         loss_func = nn.CrossEntropyLoss(reduction='mean')
#
#         beta = 0.05
#         # training and testing
#         for epoch in range(epochs):
#             print('Epoch {}/{}'.format(epoch + 1, epochs))
#             print('-' * 80)
#
#             model.train()
#
#             train_data_len = len(train_loader)  # train_data.__len__() // batch_size + 1
#             total_loss = 0
#             for step, (x, (y0, y1)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
#
#                 b_x = Variable(x.to(self.device))  # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, magnifi_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 m_loss = loss_func(magnifi_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * m_loss
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # backpropagation, compute gradients
#                 optimizer.step()
#
#                 # 数据统计
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, m_preds = torch.max(magnifi_prob, 1)
#                 running_loss = loss.item()
#                 running_corrects1 = torch.sum(c_preds == b_y0.data)
#                 running_corrects2 = torch.sum(m_preds == b_y1.data)
#
#                 total_loss += running_loss
#                 print('%d / %d ==> Total Loss: %.4f | Cancer Acc: %.4f | Magnifi Acc: %.4f '
#                       % (step, train_data_len, running_loss, running_corrects1.double() / b_x.size(0),
#                          running_corrects2.double() / b_x.size(0)))
#
#             scheduler.step(total_loss)
#
#             running_loss = 0.0
#             running_corrects1 = 0
#             running_corrects2 = 0
#             model.eval()
#             for x, (y0, y1) in test_loader:
#                 b_x = Variable(x.to(self.device))  # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, magnifi_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 m_loss = loss_func(magnifi_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * m_loss
#
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, m_preds = torch.max(magnifi_prob, 1)
#                 running_loss += loss.item() * b_x.size(0)
#                 running_corrects1 += torch.sum(c_preds == b_y0.data)
#                 running_corrects2 += torch.sum(m_preds == b_y1.data)
#
#             test_data_len = len(test_data)
#             epoch_loss = running_loss / test_data_len
#             epoch_acc_c = running_corrects1.double() / test_data_len
#             epoch_acc_m = running_corrects2.double() / test_data_len
#             torch.save(model.state_dict(),
#                        self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss,
#                                                                                       epoch_acc_c,
#                                                                                       epoch_acc_m))
#
#     def evaluate_model(self, samples_name, model_file, batch_size, max_count):
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
#         Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#         # Xtest, Ytest = Xtest[:60], Ytest[:60]  # for debug
#         test_data = Image_Dataset(Xtest, Ytest)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         model.MultiTask = False
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         test_data_len = len(test_loader)
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             # cancer_prob, magnifi_prob = model(b_x)
#             # _, cancer_preds = torch.max(cancer_prob, 1)
#             # _, magnifi_preds = torch.max(magnifi_prob, 1)
#             # for c_pred, m_pred in zip(cancer_preds.cpu().numpy(), magnifi_preds.cpu().numpy()):
#             #     predicted_tags.append((c_pred, m_pred))
#             cancer_prob = model(b_x)
#             _, cancer_preds = torch.max(cancer_prob, 1)
#             for c_pred in zip(cancer_preds.cpu().numpy()):
#                 predicted_tags.append((c_pred))
#
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#
#         Ytest = np.array(Ytest)
#         predicted_tags = np.array(predicted_tags)
#         index_x10 = Ytest[:, 1] == 0
#         index_x20 = Ytest[:, 1] == 1
#         index_x40 = Ytest[:, 1] == 2
#         print("Classification report for classifier x all:\n%s\n"
#               % (metrics.classification_report(Ytest[:,0], predicted_tags[:,0], digits=4)))
#         print("Classification report for classifier x 10:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x10,0], predicted_tags[index_x10,0], digits=4)))
#         print("Classification report for classifier x 20:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x20,0], predicted_tags[index_x20,0], digits=4)))
#         print("Classification report for classifier x 40:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x40,0], predicted_tags[index_x40,0], digits=4)))
#         # print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest[:,0], predicted_tags[:,0]))
#
#     def load_pretrained_model_on_predict(self):
#         net_file = {
#             "se_densenet_22_x_256": "se_densenet_22_x_256-cp-0022-0.0908-0.9642-0.9978.pth",
#         }
#
#         model_code = "{}_{}".format(self.model_name, self.patch_type)
#         model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[model_code])
#         model = self.load_model(model_file=model_file)
#
#         if self.patch_type in ["x_256", "msc_256"]:
#             # 关闭多任务的其它输出
#             model.MultiTask = False
#
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         return model
#
#     ###############################################################################################################
#     # Multiple scale combination (MSC)
#     ###############################################################################################################
# class MSC_Classifier(BaseClassifier):
#     def __init__(self, params, model_name, patch_type, **kwargs):
#         super(MSC_Classifier, self).__init__(params, model_name, patch_type, **kwargs)
#
#         self.num_classes = (2, 3)
#         self.image_size = 256
#
#     def load_msc_data(self, samples_name_dict):
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtrain10, Ytrain10 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtrain20, Ytrain20 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtrain40, Ytrain40 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         # Xtrain10, Xtrain20, Xtrain40, Ytrain10 = Xtrain10[:40], Xtrain20[:40],Xtrain40[:40],Ytrain10[:40] # for debug
#         train_data = Image_Dataset_MSC(Xtrain10, Xtrain20, Xtrain40, Ytrain10)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtest10, Ytest10 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtest20, Ytest20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtest40, Ytest40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         # Xtest10, Xtest20, Xtest40, Ytest10 = Xtest10[:60], Xtest20[:60], Xtest40[:60], Ytest10[:60]  # for debug
#         test_data = Image_Dataset_MSC(Xtest10, Xtest20, Xtest40, Ytest10)
#         return train_data, test_data
#
#     def create_initial_model(self):
#         if self.model_name == "se_densenet_c9_22":
#             model = self.create_se_densenet_c9(depth=22, num_init_features=36)
#         elif self.model_name == "se_densenet_c9_40":
#             model = self.create_se_densenet_c9(depth=40, num_init_features=54)
#         return model
#
#     def train_model(self, samples_name, augment_func, batch_size, epochs):
#         train_data, test_data = self.load_msc_data(samples_name)
#
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size,
#                                        shuffle=True, num_workers=self.NUM_WORKERS)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         # summary(model, torch.zeros((1, 9, self.image_size, self.image_size)), show_input=True)
#
#         model.to(self.device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
#         # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
#         loss_func = nn.CrossEntropyLoss(reduction='mean')
#
#         beta = 0.5
#         # training and testing
#         for epoch in range(epochs):
#             print('Epoch {}/{}'.format(epoch + 1, epochs))
#             print('-' * 80)
#
#             model.train()
#
#             train_data_len = len(train_loader)  # train_data.__len__() // batch_size + 1
#             total_loss = 0
#             for step, (x, (y0, y1)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
#
#                 b_x = Variable(x.to(self.device))  # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, edge_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 e_loss = loss_func(edge_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * e_loss
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # backpropagation, compute gradients
#                 optimizer.step()
#
#                 # 数据统计
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, e_preds = torch.max(edge_prob, 1)
#                 running_loss = loss.item()
#                 running_corrects1 = torch.sum(c_preds == b_y0.data)
#                 running_corrects2 = torch.sum(e_preds == b_y1.data)
#
#                 total_loss += running_loss
#                 if step % 5 == 0:
#                     print('%d / %d ==> Total Loss: %.4f | Cancer Acc: %.4f | Edge Acc: %.4f '
#                           % (step, train_data_len, running_loss, running_corrects1.double() / b_x.size(0),
#                              running_corrects2.double() / b_x.size(0)))
#
#             scheduler.step(total_loss)
#
#             running_loss = 0.0
#             running_corrects1 = 0
#             running_corrects2 = 0
#             model.eval()
#             for x, (y0, y1) in test_loader:
#                 b_x = Variable(x.to(self.device))  # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, edge_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 e_loss = loss_func(edge_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * e_loss
#
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, e_preds = torch.max(edge_prob, 1)
#                 running_loss += loss.item() * b_x.size(0)
#                 running_corrects1 += torch.sum(c_preds == b_y0.data)
#                 running_corrects2 += torch.sum(e_preds == b_y1.data)
#
#             test_data_len = len(test_data)
#             epoch_loss = running_loss / test_data_len
#             epoch_acc_c = running_corrects1.double() / test_data_len
#             epoch_acc_m = running_corrects2.double() / test_data_len
#             torch.save(model.state_dict(),
#                        self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss,
#                                                                                       epoch_acc_c,
#                                                                                       epoch_acc_m))
#
#     def evaluate_model_msc(self, samples_name_dict=None, batch_size=100):
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtest10, Ytest10 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtest20, Ytest20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtest40, Ytest40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         # Xtest10, Xtest20, Xtest40, Ytest10 = Xtest10[:60], Xtest20[:60], Xtest40[:60], Ytest10[:60]  # for debug
#         test_data = Image_Dataset_MSC(Xtest10, Xtest20, Xtest40, Ytest10)
#
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         model.MultiTask = True
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         test_data_len = len(test_loader)
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             # cancer_prob = model(b_x)
#             # _, cancer_preds = torch.max(cancer_prob, 1)
#             # for c_pred in zip(cancer_preds.cpu().numpy()):
#             #     predicted_tags.append((c_pred))
#             cancer_prob, edge_prob = model(b_x)
#             _, cancer_preds = torch.max(cancer_prob, 1)
#             _, edge_preds = torch.max(edge_prob, 1)
#             for c_pred, m_pred in zip(cancer_preds.cpu().numpy(), edge_preds.cpu().numpy()):
#                 predicted_tags.append((c_pred, m_pred))
#
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#
#         # Ytest = np.array(Ytest10[:60]) # for debug
#         Ytest = np.array(Ytest10)
#         predicted_tags = np.array(predicted_tags)
#
#         print("Classification report for classifier (normal, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:, 0], predicted_tags[:, 0], digits=4)))
#         print("Classification report for classifier (normal, edge, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:, 1], predicted_tags[:, 1], digits=4)))
#
#     def evaluate_model(self, samples_name, model_file, batch_size, max_count):
#         assert isinstance(samples_name, dict), "samples_name maust be a dict."
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name[10])
#         Xtest10, Ytest10 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name[20])
#         Xtest20, Ytest20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name[40])
#         Xtest40, Ytest40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         # Xtest10, Xtest20, Xtest40, Ytest10 = Xtest10[:60], Xtest20[:60], Xtest40[:60], Ytest10[:60]  # for debug
#         test_data = Image_Dataset_MSC(Xtest10, Xtest20, Xtest40, Ytest10)
#
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         model.MultiTask = True
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         test_data_len = len(test_loader)
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             # cancer_prob = model(b_x)
#             # _, cancer_preds = torch.max(cancer_prob, 1)
#             # for c_pred in zip(cancer_preds.cpu().numpy()):
#             #     predicted_tags.append((c_pred))
#             cancer_prob, edge_prob = model(b_x)
#             _, cancer_preds = torch.max(cancer_prob, 1)
#             _, edge_preds = torch.max(edge_prob, 1)
#             for c_pred, m_pred in zip(cancer_preds.cpu().numpy(), edge_preds.cpu().numpy()):
#                 predicted_tags.append((c_pred, m_pred))
#
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#
#         # Ytest = np.array(Ytest10[:60]) # for debug
#         Ytest = np.array(Ytest10)
#         predicted_tags = np.array(predicted_tags)
#
#         print("Classification report for classifier (normal, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:,0], predicted_tags[:,0], digits=4)))
#         print("Classification report for classifier (normal, edge, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:,1], predicted_tags[:,1], digits=4)))
#
#     def predict_on_batch(self, src_img, patch_size, seeds_scale, seeds, batch_size):
#         '''
#         预测在种子点提取的图块
#         :param src_img: 切片图像
#         :param patch_size: 图块大小
#         :param seeds_scale: 种子点的倍镜数
#         :param seeds: 种子点的集合
#         :return: 预测结果与概率的
#         '''
#         assert self.patch_type == "msc_256", "Only accept a model based on multiple scales"
#
#         if self.model is None:
#             self.model = self.load_pretrained_model_on_predict(self.patch_type)
#             self.model.to(self.device)
#             self.model.eval()
#
#         # self.model.MultiTask = True
#
#         seeds_itor = get_image_blocks_msc_itor(src_img, seeds_scale, seeds, patch_size, patch_size, batch_size)
#
#         len_seeds = len(seeds)
#         data_len = len(seeds) // batch_size
#         if len_seeds % batch_size > 0:
#             data_len += 1
#
#         results = []
#
#         for step, x in enumerate(seeds_itor):
#             b_x = Variable(x.to(self.device))
#
#             output = self.model(b_x)
#             # cancer_prob, edge_prob = self.model(b_x)
#             output_softmax = nn.functional.softmax(output)
#             probs, preds = torch.max(output_softmax, 1)
#             for prob, pred in zip(probs.cpu().numpy(), preds.cpu().numpy()):
#                 if pred == 1:
#                     results.append(prob)
#                 else:
#                     results.append(1 - prob)
#             # for prob, pred, three_prob in zip(probs.cpu().numpy(), preds.cpu().numpy(), output_softmax.cpu().numpy()):
#             #     cancer_edge_prob = three_prob[1] + three_prob[-1]
#             #     if pred == 2:
#             #         results.append((1, cancer_edge_prob))
#             #     elif pred == 1:
#             #         if three_prob[-1] > 10 * three_prob[0]:
#             #             results.append((1, cancer_edge_prob))
#             #         else:
#             #             results.append((0, 1 - three_prob[-1]))
#             #     else:
#             #         results.append((0, 1 - three_prob[-1]))
#             # for three_prob in output_softmax.cpu().numpy():
#             #     if three_prob[-1] > three_prob[0]:
#             #         results.append((1, three_prob[-1] + three_prob[1]))
#             #     else:
#             #         results.append((0, three_prob[0] + three_prob[1]))
#
#             print('predicting => %d / %d ' % (step + 1, data_len))
#
#         return results
#
#     def load_pretrained_model_on_predict(self):
#         net_file = {
#             "se_densenet_c9_22_msc_256":  "se_densenet_c9_22_msc_256_0030-0.2319-0.9775-0.6928.pth",
#         }
#
#         model_code = "{}_{}".format(self.model_name, self.patch_type)
#         model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[model_code])
#         model = self.load_model(model_file=model_file)
#
#         if self.patch_type in ["x_256", "msc_256"]:
#             # 关闭多任务的其它输出
#             model.MultiTask = False
#
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         return model



# class CNN_Classifier(BaseClassifier):
#
#     # def create_densenet(self, depth):
#     #     '''
#     #     生成指定深度的Densenet
#     #     :param depth: 深度
#     #     :return: 网络模型
#     #     '''
#     #     # Get densenet configuration
#     #     if (depth - 4) % 3:
#     #         raise Exception('Invalid depth')
#     #     block_config = [(depth - 4) // 6 for _ in range(3)]
#     #
#     #     if self.patch_type in ["cifar10", "cifar100"]:  # 32x32
#     #         # Models
#     #         model = DenseNet(
#     #             growth_rate=12,
#     #             block_config=block_config,
#     #             num_classes=self.num_classes,
#     #             small_inputs=True, # 32 x 32的图片为True
#     #             gvp_out_size=1,
#     #             efficient=True,
#     #         )
#     #     elif self.patch_type in ["500_128", "2000_256", "4000_256", "x_256"]: # 256 x 256
#     #         # Models
#     #         model = DenseNet(
#     #             growth_rate=12,
#     #             block_config=block_config,
#     #             num_classes=self.num_classes,
#     #             small_inputs=False, # 32 x 32的图片为True
#     #             gvp_out_size=1,
#     #             efficient=True,
#     #         )
#     #     return  model
#     #
#     # def create_se_densenet(self, depth):
#     #     # Get densenet configuration
#     #     if (depth - 4) % 3:
#     #         raise Exception('Invalid depth')
#     #     block_config = [(depth - 4) // 6 for _ in range(3)]
#     #
#     #     # Models
#     #     model = SEDenseNet(
#     #         growth_rate=12,
#     #         block_config=block_config,
#     #         num_classes=self.num_classes,
#     #         gvp_out_size=1,
#     #     )
#     #     return  model
#
#     def create_se_densenet_c9(self, depth, num_init_features):
#         # Get densenet configuration
#         if (depth - 4) % 3:
#             raise Exception('Invalid depth')
#         block_config = [(depth - 4) // 6 for _ in range(3)]
#
#         # Models
#         model = SEDenseNet_C9(
#             growth_rate=12,
#             block_config=block_config,
#             num_init_features=num_init_features,
#             num_classes=self.num_classes,
#             gvp_out_size=1,
#         )
#         return  model
#
#     def create_initial_model(self):
#         '''
#         生成初始化的模型
#         :return:网络模型
#         '''
#         if self.model_name == "simple_cnn":
#             model = Simple_CNN(self.num_classes, self.image_size)
#         elif self.model_name == "densenet_22":
#             model = self.create_densenet(depth=22)
#         elif self.model_name == "se_densenet_22":
#             model = self.create_se_densenet(depth=22)
#         elif self.model_name =="se_densenet_40":
#             model=self.create_se_densenet(depth=40)
#         elif self.model_name == "se_densenet_c9_22":
#             model = self.create_se_densenet_c9(depth=22, num_init_features=36)
#         elif self.model_name == "se_densenet_c9_40":
#             model = self.create_se_densenet_c9(depth=40, num_init_features=54)
#         return model
#
#     def load_model(self, model_file = None):
#         '''
#         加载模型
#         :param model_file: 模型文件
#         :return: 网络模型
#         '''
#         if model_file is not None:
#             print("loading >>> ", model_file, " ...")
#             model = torch.load(model_file)
#             return model
#         else:
#             checkpoint_dir = self.model_root
#             if (not os.path.exists(checkpoint_dir)):
#                 os.makedirs(checkpoint_dir)
#
#             latest = latest_checkpoint(checkpoint_dir)
#             if latest is not None:
#                 print("loading >>> ", latest, " ...")
#                 model = torch.load(latest)
#             else:
#                 model = self.create_initial_model()
#             return model
#
#     def train_model(self, samples_name = None, augment_func = None, batch_size=100, epochs=20):
#         '''
#         训练模型
#         :param samples_name: 自制训练集的代号
#         :param batch_size: 每批的图片数量
#         :param epochs:epoch数量
#         :return:
#         '''
#         if self.patch_type in ["cifar10", "cifar100"]:
#             train_data, test_data = self.load_cifar_data(self.patch_type)
#         elif self.patch_type in ["500_128", "2000_256", "4000_256"]:
#             train_data, test_data = self.load_custom_data(samples_name, augment_func = augment_func)
#
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=self.NUM_WORKERS)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         print(model)
#         if self.use_GPU:
#             model.cuda()
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #学习率为0.01的学习器
#         # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                factor=0.5)  # mode为min，则loss不下降学习率乘以factor，max则反之
#         loss_func = nn.CrossEntropyLoss()
#
#         # training and testing
#         for epoch in range(epochs):
#             print('Epoch {}/{}'.format(epoch + 1, epochs))
#             print('-' * 80)
#
#             model.train()
#             # 开始训练
#             train_data_len = len(train_loader)
#             total_loss = 0
#             for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
#                 if self.use_GPU:
#                     b_x = Variable(x).cuda()  # batch x
#                     b_y = Variable(y).cuda()  # batch y
#                 else:
#                     b_x = Variable(x)  # batch x
#                     b_y = Variable(y)  # batch y
#
#                 output = model(b_x)  # cnn output
#                 loss = loss_func(output, b_y)  # cross entropy loss
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # backpropagation, compute gradients
#                 optimizer.step()
#
#                 # 数据统计
#                 _, preds = torch.max(output, 1)
#
#                 running_loss = loss.item()
#                 running_corrects = torch.sum(preds == b_y.data)
#                 total_loss += running_loss
#                 print('%d / %d ==> Loss: %.4f | Acc: %.4f '
#                       % (step, train_data_len, running_loss, running_corrects.double()/b_x.size(0)))
#
#             scheduler.step(total_loss)
#
#             running_loss=0.0
#             running_corrects=0
#             model.eval()
#             # 开始评估
#             for x, y in test_loader:
#                 if self.use_GPU:
#                     b_x = Variable(x).cuda()  # batch x
#                     b_y = Variable(y).cuda()  # batch y
#                 else:
#                     b_x = Variable(x)  # batch x
#                     b_y = Variable(y)  # batch y
#
#                 output = model(b_x)
#                 loss = loss_func(output, b_y)
#
#                 _, preds = torch.max(output, 1)
#                 running_loss += loss.item() * b_x.size(0)
#                 running_corrects += torch.sum(preds == b_y.data)
#
#             test_data_len = test_data.__len__()
#             epoch_loss=running_loss / test_data_len
#             epoch_acc=running_corrects.double() / test_data_len
#
#             torch.save(model, self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch+1, epoch_loss, epoch_acc))
#
#         return
#
#     def evaluate_model(self, samples_name=None, model_file=None, batch_size=100, max_count = None):
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH[samples_name[0]], samples_name[1])
#         Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH[samples_name[0]], test_list)
#         if max_count is not None:
#             Xtest, Ytest = Xtest[:max_count], Ytest[:max_count]  # for debug
#
#         test_data = Image_Dataset(Xtest, Ytest, norm = self.normal_func)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#         # image_itor = get_image_file_batch_normalize_itor(Xtest, Ytest, batch_size, self.normal_func)
#
#         model = self.load_model(model_file=model_file)
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         len_y = len(Ytest)
#         if len_y % batch_size == 0:
#             test_data_len = len_y // batch_size
#         else:
#             test_data_len = len_y // batch_size + 1
#
#         # for step, (x, _) in enumerate(image_itor):
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             output = model(b_x) # model最后不包括一个softmax层
#             output_softmax = nn.functional.softmax(output, dim=1)
#             probs, preds = torch.max(output_softmax, 1)
#
#             predicted_tags.extend(preds.cpu().numpy())
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#             probs = probs.cpu().numpy()
#             mean = np.mean(probs)
#             std = np.std(probs)
#             print("mean of prob = ", mean, std)
#
#         Ytest = np.array(Ytest)
#         predicted_tags = np.array(predicted_tags)
#         print("Classification report for classifier :\n%s\n"
#               % (metrics.classification_report(Ytest, predicted_tags, digits=4)))
#
#     def load_cifar_data(self, patch_type):
#         '''
#         加载cifar数量
#         :param patch_type: cifar 数据的代号
#         :return:
#         '''
#         data_root = os.path.join(os.path.expanduser('~'), '.keras/datasets/') # 共用Keras下载的数据
#
#         if patch_type == "cifar10":
#             train_data = torchvision.datasets.cifar.CIFAR10(
#                 root=data_root,  # 保存或者提取位置
#                 train=True,  # this is training data
#                 transform=torchvision.transforms.ToTensor(),
#                 download = False
#             )
#             test_data = torchvision.datasets.cifar.CIFAR10(root=data_root, train=False,
#                                                    transform=torchvision.transforms.ToTensor())
#             return train_data, test_data
#
#     def load_custom_data(self, samples_name, augment_func = None):
#         '''
#         从图片的列表文件中加载数据，到Sequence中
#         :param samples_name: 列表文件的代号
#         :return:用于train和test的两个Sequence
#         '''
#         patch_root = self._params.PATCHS_ROOT_PATH[samples_name[0]]
#         sample_filename = samples_name[1]
#         train_list = "{}/{}_train.txt".format(patch_root, sample_filename)
#         test_list = "{}/{}_test.txt".format(patch_root, sample_filename)
#
#         Xtrain, Ytrain = read_csv_file(patch_root, train_list)
#         # Xtrain, Ytrain = Xtrain[:40], Ytrain[:40] # for debug
#         train_data = Image_Dataset(Xtrain, Ytrain, transform = None, augm = augment_func)
#
#         Xtest, Ytest = read_csv_file(patch_root, test_list)
#         # Xtest, Ytest = Xtest[:60], Ytest[:60]  # for debug
#         test_data = Image_Dataset(Xtest, Ytest)
#         return  train_data, test_data
#
#     def load_pretrained_model_on_predict(self, patch_type):
#         '''
#         加载已经训练好的存盘网络文件
#         :param patch_type: 分类器处理图块的类型
#         :return: 网络模型
#         '''
#         # net_file = {"500_128":  "densenet_22_500_128_cp-0017-0.2167-0.9388.pth",
#         #             "2000_256": "densenet_22_2000_256-cp-0019-0.0681-0.9762.pth",
#         #             "4000_256": "densenet_22_4000_256-cp-0019-0.1793-0.9353.pth",
#         #             "x_256" :   "se_densenet_22_x_256-cp-0022-0.0908-0.9642-0.9978.pth",
#         #             "msc_256":  "se_densenet_c9_22_msc_256_0030-0.2319-0.9775-0.6928.pth",
#         #             }
#         net_file = {
#                     "4000_256": "simple_cnn_40_256_cp-0003-0.0742-0.9743.pth",
#                     }
#
#         model_file = "{}/models/pytorch/trained/{}".format(self._params.PROJECT_ROOT, net_file[patch_type])
#         model = self.load_model(model_file=model_file)
#
#         if patch_type in ["x_256", "msc_256"]:
#             # 关闭多任务的其它输出
#             model.MultiTask = False
#
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         return model
#
#
#     def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size):
#         '''
#         预测在种子点提取的图块
#         :param src_img: 切片图像
#         :param scale: 提取图块的倍镜数
#         :param patch_size: 图块大小
#         :param seeds: 种子点的集合
#         :return: 预测结果与概率
#         '''
#         # seeds_itor = get_image_blocks_itor(src_img, scale, seeds, patch_size, patch_size, batch_size,
#         #                                    normalization=self.normal_func)
#         seeds_itor = get_image_blocks_batch_normalize_itor(src_img, scale, seeds, patch_size, patch_size, batch_size,
#                                            normalization=self.normal_func)
#
#         if self.model is None:
#             self.model = self.load_pretrained_model_on_predict(self.patch_type)
#             self.model.to(self.device)
#             self.model.eval()
#
#         len_seeds = len(seeds)
#         data_len = len(seeds) // batch_size
#         if len_seeds % batch_size > 0:
#             data_len += 1
#
#         results = []
#
#         for step, x in enumerate(seeds_itor):
#             b_x = Variable(x.to(self.device))
#
#             output = self.model(b_x) # model最后不包括一个softmax层
#             output_softmax = nn.functional.softmax(output, dim =1)
#             probs, preds = torch.max(output_softmax, 1)
#             for prob, pred in zip(probs.cpu().numpy(), preds.cpu().numpy()):
#                 results.append((pred, prob))
#             print('predicting => %d / %d ' % (step + 1, data_len))
#
#             probs = probs.cpu().numpy()
#             mean = np.mean(probs)
#             std = np.std(probs)
#             print("mean of prob = ", mean, std)
#
#         return results
#
# ######################################################################################################################
#
# ############       multi task            #########
#
# ######################################################################################################################
#     def train_model_multi_task(self, samples_name=None, batch_size=100, epochs=20):
#
#         train_data, test_data = self.load_custom_data(samples_name)
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size,
#                                        shuffle=True, num_workers = self.NUM_WORKERS)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers = self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         summary(model, input_size=(3, self.image_size, self.image_size), device="cpu")
#
#         model.to(self.device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
#         # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
#         loss_func = nn.CrossEntropyLoss(reduction='mean')
#
#         beta = 0.05
#         # training and testing
#         for epoch in range(epochs):
#             print('Epoch {}/{}'.format(epoch + 1, epochs))
#             print('-' * 80)
#
#             model.train()
#
#             train_data_len = len(train_loader) # train_data.__len__() // batch_size + 1
#             total_loss = 0
#             for step, (x, (y0, y1)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
#
#                 b_x = Variable(x.to(self.device)) # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, magnifi_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 m_loss = loss_func(magnifi_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * m_loss
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # backpropagation, compute gradients
#                 optimizer.step()
#
#                 # 数据统计
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, m_preds = torch.max(magnifi_prob, 1)
#                 running_loss = loss.item()
#                 running_corrects1 = torch.sum(c_preds == b_y0.data)
#                 running_corrects2 = torch.sum(m_preds == b_y1.data)
#
#                 total_loss += running_loss
#                 print('%d / %d ==> Total Loss: %.4f | Cancer Acc: %.4f | Magnifi Acc: %.4f '
#                       % (step, train_data_len, running_loss, running_corrects1.double() / b_x.size(0),
#                          running_corrects2.double() / b_x.size(0)))
#
#             scheduler.step(total_loss)
#
#             running_loss = 0.0
#             running_corrects1 = 0
#             running_corrects2 = 0
#             model.eval()
#             for x, (y0, y1) in test_loader:
#
#                 b_x = Variable(x.to(self.device)) # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, magnifi_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 m_loss = loss_func(magnifi_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * m_loss
#
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, m_preds = torch.max(magnifi_prob, 1)
#                 running_loss += loss.item() * b_x.size(0)
#                 running_corrects1 += torch.sum(c_preds == b_y0.data)
#                 running_corrects2 += torch.sum(m_preds == b_y1.data)
#
#             test_data_len = len(test_data)
#             epoch_loss = running_loss / test_data_len
#             epoch_acc_c = running_corrects1.double() / test_data_len
#             epoch_acc_m = running_corrects2.double() / test_data_len
#             torch.save(model,
#                        self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss, epoch_acc_c,
#                                                                                epoch_acc_m))
#
#
#     def evaluate_model_multi_task(self, samples_name=None, batch_size=100):
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
#         Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#         # Xtest, Ytest = Xtest[:60], Ytest[:60]  # for debug
#         test_data = Image_Dataset(Xtest, Ytest)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         model.MultiTask = False
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         test_data_len = len(test_loader)
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             # cancer_prob, magnifi_prob = model(b_x)
#             # _, cancer_preds = torch.max(cancer_prob, 1)
#             # _, magnifi_preds = torch.max(magnifi_prob, 1)
#             # for c_pred, m_pred in zip(cancer_preds.cpu().numpy(), magnifi_preds.cpu().numpy()):
#             #     predicted_tags.append((c_pred, m_pred))
#             cancer_prob = model(b_x)
#             _, cancer_preds = torch.max(cancer_prob, 1)
#             for c_pred in zip(cancer_preds.cpu().numpy()):
#                 predicted_tags.append((c_pred))
#
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#
#         Ytest = np.array(Ytest)
#         predicted_tags = np.array(predicted_tags)
#         index_x10 = Ytest[:, 1] == 0
#         index_x20 = Ytest[:, 1] == 1
#         index_x40 = Ytest[:, 1] == 2
#         print("Classification report for classifier x all:\n%s\n"
#               % (metrics.classification_report(Ytest[:,0], predicted_tags[:,0], digits=4)))
#         print("Classification report for classifier x 10:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x10,0], predicted_tags[index_x10,0], digits=4)))
#         print("Classification report for classifier x 20:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x20,0], predicted_tags[index_x20,0], digits=4)))
#         print("Classification report for classifier x 40:\n%s\n"
#               % (metrics.classification_report(Ytest[index_x40,0], predicted_tags[index_x40,0], digits=4)))
#         # print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest[:,0], predicted_tags[:,0]))
#
#
#     def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
#         '''
#         预测在种子点提取的图块
#         :param src_img: 切片图像
#         :param scale_tuple: 提取图块的倍镜数的tuple
#         :param patch_size: 图块大小
#         :param seeds_scale: 种子点的倍镜数
#         :param seeds: 种子点的集合
#         :return: 预测结果与概率的
#         '''
#         assert self.patch_type == "x_256", "Only accept a model based on multiple scales"
#
#         if self.model is None:
#             self.model = self.load_pretrained_model_on_predict(self.patch_type)
#             self.model.to(self.device)
#             self.model.eval()
#
#         scale_tuple = (10, 20, 40)
#         len_seeds = len(seeds)
#         len_scale = len(scale_tuple)
#
#         data_len = len(seeds) // batch_size
#         if len_seeds % batch_size > 0:
#             data_len += 1
#
#         multi_results = np.empty((len_seeds, len_scale))
#
#         for index, extract_scale in enumerate(scale_tuple):
#             high_seeds = transform_coordinate(0, 0, seeds_scale, seeds_scale, extract_scale, seeds)
#             seeds_itor = get_image_blocks_itor(src_img, extract_scale, high_seeds, patch_size, patch_size, batch_size)
#
#             results = []
#             for step, x in enumerate(seeds_itor):
#                 b_x = Variable(x.to(self.device))
#
#                 output = self.model(b_x)
#                 output_softmax = nn.functional.softmax(output)
#                 probs, preds = torch.max(output_softmax, 1)
#                 for prob, pred in zip(probs.cpu().numpy(), preds.cpu().numpy()):
#                     # results.append((pred,prob))
#                     if pred == 1:
#                         results.append(prob)
#                     else:
#                         results.append(1 - prob)
#
#                 print('scale = %d, predicting => %d / %d ' % (extract_scale, step + 1, data_len))
#
#             multi_results[:,index] = results
#
#         return np.max(multi_results, axis=1)   # 0.93
#         # return np.mean(multi_results, axis=1)  # 0.88
#         # return np.min(multi_results, axis=1)  # 0.58
#
#     ###############################################################################################################
#     # Multiple scale combination (MSC)
#     ###############################################################################################################
#     def load_msc_data(self, samples_name_dict):
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtrain10, Ytrain10 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtrain20, Ytrain20 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtrain40, Ytrain40 = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
#
#         # Xtrain10, Xtrain20, Xtrain40, Ytrain10 = Xtrain10[:40], Xtrain20[:40],Xtrain40[:40],Ytrain10[:40] # for debug
#         train_data = Image_Dataset_MSC(Xtrain10, Xtrain20, Xtrain40, Ytrain10)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtest10, Ytest10 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtest20, Ytest20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtest40, Ytest40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         # Xtest10, Xtest20, Xtest40, Ytest10 = Xtest10[:60], Xtest20[:60], Xtest40[:60], Ytest10[:60]  # for debug
#         test_data = Image_Dataset_MSC(Xtest10, Xtest20, Xtest40, Ytest10)
#         return train_data, test_data
#
#     def train_model_msc(self, samples_name=None, batch_size=100, epochs=20):
#
#         train_data, test_data = self.load_msc_data(samples_name)
#
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size,
#                                        shuffle=True, num_workers = self.NUM_WORKERS)
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers = self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         summary(model, input_size=(9, self.image_size, self.image_size), device="cpu")
#
#         model.to(self.device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 学习率为0.01的学习器
#         # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每过30个epoch训练，学习率就乘gamma
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
#         loss_func = nn.CrossEntropyLoss(reduction='mean')
#
#         beta = 0.5
#         # training and testing
#         for epoch in range(epochs):
#             print('Epoch {}/{}'.format(epoch + 1, epochs))
#             print('-' * 80)
#
#             model.train()
#
#             train_data_len = len(train_loader) # train_data.__len__() // batch_size + 1
#             total_loss = 0
#             for step, (x, (y0, y1)) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
#
#                 b_x = Variable(x.to(self.device)) # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, edge_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 e_loss = loss_func(edge_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * e_loss
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # backpropagation, compute gradients
#                 optimizer.step()
#
#                 # 数据统计
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, e_preds = torch.max(edge_prob, 1)
#                 running_loss = loss.item()
#                 running_corrects1 = torch.sum(c_preds == b_y0.data)
#                 running_corrects2 = torch.sum(e_preds == b_y1.data)
#
#                 total_loss += running_loss
#                 print('%d / %d ==> Total Loss: %.4f | Cancer Acc: %.4f | Edge Acc: %.4f '
#                       % (step, train_data_len, running_loss, running_corrects1.double() / b_x.size(0),
#                          running_corrects2.double() / b_x.size(0)))
#
#             scheduler.step(total_loss)
#
#             running_loss = 0.0
#             running_corrects1 = 0
#             running_corrects2 = 0
#             model.eval()
#             for x, (y0, y1) in test_loader:
#
#                 b_x = Variable(x.to(self.device)) # batch x
#                 b_y0 = Variable(y0.to(self.device))  # batch y0
#                 b_y1 = Variable(y1.to(self.device))  # batch y1
#
#                 cancer_prob, edge_prob = model(b_x)
#                 c_loss = loss_func(cancer_prob, b_y0)  # cross entropy loss
#                 e_loss = loss_func(edge_prob, b_y1)  # cross entropy loss
#                 loss = (1 - beta) * c_loss + beta * e_loss
#
#                 _, c_preds = torch.max(cancer_prob, 1)
#                 _, e_preds = torch.max(edge_prob, 1)
#                 running_loss += loss.item() * b_x.size(0)
#                 running_corrects1 += torch.sum(c_preds == b_y0.data)
#                 running_corrects2 += torch.sum(e_preds == b_y1.data)
#
#             test_data_len = len(test_data)
#             epoch_loss = running_loss / test_data_len
#             epoch_acc_c = running_corrects1.double() / test_data_len
#             epoch_acc_m = running_corrects2.double() / test_data_len
#             torch.save(model,
#                        self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss, epoch_acc_c,
#                                                                                epoch_acc_m))
#
#     def evaluate_model_msc(self, samples_name_dict=None, batch_size=100):
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[10])
#         Xtest10, Ytest10 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[20])
#         Xtest20, Ytest20 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name_dict[40])
#         Xtest40, Ytest40 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
#
#         # Xtest10, Xtest20, Xtest40, Ytest10 = Xtest10[:60], Xtest20[:60], Xtest40[:60], Ytest10[:60]  # for debug
#         test_data = Image_Dataset_MSC(Xtest10, Xtest20, Xtest40, Ytest10)
#
#         test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
#                                       shuffle=False, num_workers=self.NUM_WORKERS)
#
#         model = self.load_model(model_file=None)
#         model.MultiTask = True
#         # 关闭求导，节约大量的显存
#         for param in model.parameters():
#             param.requires_grad = False
#         print(model)
#
#         model.to(self.device)
#         model.eval()
#
#         predicted_tags = []
#         test_data_len = len(test_loader)
#         for step, (x, _) in enumerate(test_loader):
#             b_x = Variable(x.to(self.device))  # batch x
#
#             # cancer_prob = model(b_x)
#             # _, cancer_preds = torch.max(cancer_prob, 1)
#             # for c_pred in zip(cancer_preds.cpu().numpy()):
#             #     predicted_tags.append((c_pred))
#             cancer_prob, edge_prob = model(b_x)
#             _, cancer_preds = torch.max(cancer_prob, 1)
#             _, edge_preds = torch.max(edge_prob, 1)
#             for c_pred, m_pred in zip(cancer_preds.cpu().numpy(), edge_preds.cpu().numpy()):
#                 predicted_tags.append((c_pred, m_pred))
#
#             print('predicting => %d / %d ' % (step + 1, test_data_len))
#
#         # Ytest = np.array(Ytest10[:60]) # for debug
#         Ytest = np.array(Ytest10)
#         predicted_tags = np.array(predicted_tags)
#
#         print("Classification report for classifier (normal, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:,0], predicted_tags[:,0], digits=4)))
#         print("Classification report for classifier (normal, edge, cancer):\n%s\n"
#               % (metrics.classification_report(Ytest[:,1], predicted_tags[:,1], digits=4)))
#
#     def predict_msc(self, src_img, patch_size, seeds_scale, seeds, batch_size):
#         '''
#         预测在种子点提取的图块
#         :param src_img: 切片图像
#         :param patch_size: 图块大小
#         :param seeds_scale: 种子点的倍镜数
#         :param seeds: 种子点的集合
#         :return: 预测结果与概率的
#         '''
#         assert self.patch_type == "msc_256", "Only accept a model based on multiple scales"
#
#         if self.model is None:
#             self.model = self.load_pretrained_model_on_predict(self.patch_type)
#             self.model.to(self.device)
#             self.model.eval()
#
#         # self.model.MultiTask = True
#
#         seeds_itor = get_image_blocks_msc_itor(src_img, seeds_scale, seeds, patch_size, patch_size, batch_size)
#
#         len_seeds = len(seeds)
#         data_len = len(seeds) // batch_size
#         if len_seeds % batch_size > 0:
#             data_len += 1
#
#         results = []
#
#         for step, x in enumerate(seeds_itor):
#             b_x = Variable(x.to(self.device))
#
#             output = self.model(b_x)
#             # cancer_prob, edge_prob = self.model(b_x)
#             output_softmax = nn.functional.softmax(output)
#             probs, preds = torch.max(output_softmax, 1)
#             for prob, pred in zip(probs.cpu().numpy(), preds.cpu().numpy()):
#                 if pred == 1:
#                     results.append(prob)
#                 else:
#                     results.append(1 - prob)
#             # for prob, pred, three_prob in zip(probs.cpu().numpy(), preds.cpu().numpy(), output_softmax.cpu().numpy()):
#             #     cancer_edge_prob = three_prob[1] + three_prob[-1]
#             #     if pred == 2:
#             #         results.append((1, cancer_edge_prob))
#             #     elif pred == 1:
#             #         if three_prob[-1] > 10 * three_prob[0]:
#             #             results.append((1, cancer_edge_prob))
#             #         else:
#             #             results.append((0, 1 - three_prob[-1]))
#             #     else:
#             #         results.append((0, 1 - three_prob[-1]))
#             # for three_prob in output_softmax.cpu().numpy():
#             #     if three_prob[-1] > three_prob[0]:
#             #         results.append((1, three_prob[-1] + three_prob[1]))
#             #     else:
#             #         results.append((0, three_prob[0] + three_prob[1]))
#
#             print('predicting => %d / %d ' % (step + 1, data_len))
#
#         return results

