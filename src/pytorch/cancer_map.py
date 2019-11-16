#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-07-31'

"""
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from modelsummary import summary
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
from skimage import morphology
from skimage.morphology import square
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset

from core.util import latest_checkpoint
from pytorch.loss_function import CenterLoss, LGMLoss, LGMLoss_v0
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import joblib

class CancerMapBuilder(object):
    def __init__(self, params, extract_scale, patch_size):
        self._params = params
        seeds_scale = self._params.GLOBAL_SCALE
        amplify = extract_scale / seeds_scale
        self.selem_size = int(0.5 * patch_size / amplify)

    def generating_probability_map(self, history, x1, y1, x2, y2, scale):
        '''
        生成 tumor probability heatmap
        :param history: 预测的结果（坐标，概率）
        :param x1: 检测区域的左上角x
        :param y1: 检测区域的左上角y
        :param x2: 检测区域的右下角x
        :param y2: 检测区域的右下角y
        :param scale:  上述坐标的倍镜数
        :return:
        '''
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)
        valid_area_width = xx2 - xx1
        valid_area_height = yy2 - yy1

        value = np.array(list(history.values()))
        point = list(history.keys())
        value_softmax = 1 / (1 + np.exp(-value))

        # 生成坐标网格
        grid_y, grid_x = np.mgrid[0: valid_area_height: 1, 0: valid_area_width: 1]
        cancer_map = griddata(point, value_softmax, (grid_x, grid_y), method='linear', fill_value=0)

        # cancer_map = morphology.closing(cancer_map, square(2 * self.selem_size))
        # cancer_map = morphology.dilation(cancer_map, square(self.selem_size))

        return cancer_map

    @staticmethod
    def calc_probability_threshold(history, t = 0):
        '''

        :param history: 预测的结果（坐标，概率）
        :param t: 筛选正样本用的阈值
        :return: 高低两个阈值
        '''
        value = np.array(list(history.values()))
        tag = value > t
        positive_part = np.reshape(value[tag], (-1, 1))

        p_count = len(positive_part)
        if p_count > 100:
            contamination = 0.1
            ift = IsolationForest(behaviour='new', max_samples='auto', contamination=contamination)
            y = ift.fit_predict(positive_part)
            outliers = positive_part[y == -1]
        else:
            outliers = positive_part

        clustering = KMeans(n_clusters=2, max_iter=100, tol=1e-4, random_state=None).fit(outliers)

        cluster_centers = clustering.cluster_centers_.ravel()
        dist = abs(cluster_centers[0] - cluster_centers[1])
        if cluster_centers[0] < cluster_centers[1]:
            left_id,  right_id= 0, 1
        else:
            left_id, right_id = 1, 0

        if dist >= 1.0:
            left_part = outliers[clustering.labels_ == left_id]
            right_part = outliers[clustering.labels_ == right_id]

            # low_thresh = np.mean(left_part)
            low_thresh = np.max(left_part)
            high_thresh = np.min(right_part)

            low_prob_thresh = 1 / (1 + np.exp(-low_thresh))
            high_prob_thresh = 1 / (1 + np.exp(-high_thresh))
            return low_prob_thresh, high_prob_thresh
        else:
            f =  0.5 * (cluster_centers[left_id] + cluster_centers[right_id])
            high_prob_thresh = 1 / (1 + np.exp(-f))
            low_prob_thresh = 1 / (1 + np.exp(-t))
            return low_prob_thresh, high_prob_thresh

    # @staticmethod
    # def calc_probability_threshold(history, t = 0):
    #     value = np.array(list(history.values()))
    #     tag = value > t
    #     positive_part = np.reshape(value[tag], (-1, 1))
    #
    #     p_count = len(positive_part)
    #     if p_count > 40:
    #         contamination = 0.1
    #     else:
    #         return 0.5, 0.5
    #
    #     ift = IsolationForest(behaviour='new', max_samples='auto', contamination=contamination)
    #     y = ift.fit_predict(positive_part)
    #     outliers = positive_part[y == -1]
    #
    #     clustering = KMeans(n_clusters=2, max_iter=100, tol=1e-4, random_state=None).fit(outliers)
    #
    #     cluster_centers = clustering.cluster_centers_.ravel()
    #     dist = abs(cluster_centers[0] - cluster_centers[1])
    #
    #     if cluster_centers[0] < cluster_centers[1]:
    #         left_part = outliers[clustering.labels_ == 0]
    #         right_part = outliers[clustering.labels_ == 1]
    #         right_id = 1
    #     else:
    #         left_part = outliers[clustering.labels_ == 1]
    #         right_part = outliers[clustering.labels_ == 0]
    #         right_id = 0
    #
    #     low_thresh = np.min(right_part)
    #     high_thresh = cluster_centers[right_id]
    #
    #     low_prob_thresh = 1 / (1 + np.exp(-low_thresh))
    #     high_prob_thresh = 1 / (1 + np.exp(-high_thresh))
    #     return low_prob_thresh, high_prob_thresh

        # if dist >= 1.0:
        #     if cluster_centers[0] < cluster_centers[1]:
        #         left_part = outliers[clustering.labels_ == 0]
        #         right_part = outliers[clustering.labels_ == 1]
        #     else:
        #         left_part = outliers[clustering.labels_ == 1]
        #         right_part = outliers[clustering.labels_ == 0]
        #
        #     low_thresh = np.max(left_part)
        #     high_thresh = np.min(right_part)
        #
        #     low_prob_thresh = 1 / (1 + np.exp(-low_thresh))
        #     high_prob_thresh = 1 / (1 + np.exp(-high_thresh))
        #     return low_prob_thresh, high_prob_thresh
        # else:
        #     max_value = np.amax(outliers)
        #     min_value = np.amin(outliers)
        #     f =  0.5 * (cluster_centers[0] + cluster_centers[1])
        #     low_prob_thresh = 1 / (1 + np.exp(-0.5 * (f + min_value)))
        #     high_prob_thresh = 1 / (1 + np.exp(-0.5 * (f + max_value)))
        #     return low_prob_thresh, high_prob_thresh


class Slide_CNN(nn.Module):

    def __init__(self, image_size):
        super(Slide_CNN, self).__init__()

        num_classes = 2
        self.conv1 = nn.Sequential(  # input shape (1, 128, 128) or (1, 256, 256)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(32),  # activation
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(32),
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, image_size / 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(48),
            nn.MaxPool2d(kernel_size=2)  # image_size / 4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.ReLU(64),
            nn.MaxPool2d(kernel_size=2) # image_size / 8
        )
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AvgPool2d(8, stride=8)
        s = int(image_size / (8 * 8))
        feature_dim = 64 * s * s
        # self.dense = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.ReLU(256),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图成 (batch_size, .....)
        # x = self.dense(x)

        output = self.classifier(x)
        return output

class SlideFilter(object):
    def __init__(self, params, model_name, patch_type):
        self._params = params

        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS
        self.image_size = int(patch_type)
        self.num_classes = 2

        self.model_root = "{}/models/pytorch/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        self.model = None

        self.use_GPU = True
        self.device = torch.device("cuda:0" if self.use_GPU else "cpu")

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

    def create_initial_model(self):
        return Slide_CNN(self.image_size)


    def train(self, data_filename, class_weight, batch_size, loss_weight, epochs):
        '''
        滤波器 训练
        :param data_filename: 样本文件名
        :param class_weight:类权重
        :param batch_size: batch size
        :param loss_weight: center loss的权重
        :param epochs: epochs
        :return:
        '''
        filename = "{}/data/{}".format(self._params.PROJECT_ROOT, data_filename)
        D = np.load(filename, allow_pickle=True)
        X = D['x']
        Y = D['y']
        # X = np.reshape(X, (-1,1,self.image_size, self.image_size))

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, )

        # train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
        #                                             torch.from_numpy(y_train).long())
        # test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(),
        #                                            torch.from_numpy(y_test).long())
        train_data = Sparse_Image_Dataset(X_train, y_train)
        test_data = Sparse_Image_Dataset(X_test, y_test)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        summary(model, torch.zeros((1, 1, self.image_size, self.image_size)), show_input=False)

        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight)

        classifi_loss = nn.CrossEntropyLoss(weight=class_weight)
        # center_loss = CenterLoss(self.num_classes, 2)
        center_loss = LGMLoss_v0(self.num_classes, 2, 1.00)
        if self.use_GPU:
            model.to(self.device)
            classifi_loss.to(self.device)
            center_loss.to(self.device)

        # optimzer4nn
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # classifi_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,)
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

                output = model(b_x)  # cnn output

                # cross entropy loss + center loss
                # loss = classifi_loss(output, b_y) + loss_weight * center_loss(b_y, output)

                logits, mlogits, likelihood = center_loss(output, b_y)
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

            save_filename = self.model_root + "/{}_{}_cp-{:04d}-{:.4f}-{:.4f}.pth".format(
                self.model_name, self.patch_type,epoch+1, epoch_loss, epoch_acc)
            torch.save(model.state_dict(), save_filename)
            print("Saved ", save_filename)
        return


    def predict(self, x1, y1, x2, y2, history, batch_size):
        '''
        使用Slide Filter进行滤波过程
        :param x1: 检测区域的左上角x
        :param y1: 检测区域的左上角y
        :param x2: 检测区域的右下角x
        :param y2: 检测区域的右下角y
        :param history: 预测的结果（坐标，概率）
        :param batch_size:
        :return:
        '''
        valid_area_width = x2 - x1
        valid_area_height = y2 - y1
        cancer_map = np.zeros((valid_area_height, valid_area_width),dtype=np.float)
        for (x, y), f in history.items():
            cancer_map[y, x] = f

        # 是否转换成稀疏矩阵，以压缩内存，
        c_tag = self.image_size > 150

        X_data = []
        half_size = self.image_size // 2
        xy = []
        # prob = []
        for (x, y), f in history.items():
            sx1 = x - half_size
            sx2 = x + half_size
            sy1 = y - half_size
            sy2 = y + half_size
            if sx1 < 0 or sx2 > valid_area_width or sy1 < 0 or sy2 > valid_area_height:
                continue
            sub_m = cancer_map[sy1:sy2, sx1:sx2]

            if c_tag:
                sub_m = coo_matrix(sub_m)

            X_data.append(sub_m)
            xy.append((x, y))
            # prob.append(f)

        if self.model is None:
            self.model = self.load_model(model_file=None)

            # 关闭求导，节约大量的显存
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.to(self.device)
            self.model.eval()

        if c_tag:
            test_data = Sparse_Image_Dataset(X_data, None)
        else:
            X_data = np.array(X_data).reshape((-1,1,self.image_size,self.image_size))
            test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_data).float(),)

        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        data_len = len(test_loader)

        prediction = []
        for step, xx in enumerate(test_loader):
            if c_tag:
                b_x = Variable(xx.to(self.device))
            else:
                b_x = Variable(xx[0].to(self.device))

            output = self.model(b_x)  # model最后不包括一个softmax层

            prediction.extend(output.cpu().numpy())
            if (step % 20) == 0:
                print('predicting => %d / %d ' % (step + 1, data_len))

        new_history = dict(history)
        for (x, y), pred in zip(xy, prediction):
            new_history[(x, y)] = pred[1]

        return new_history

    def counting_history_changes(self, old_history, new_history):
        '''
        统计被翻转的预测结果的数量
        :param old_history: old predictions
        :param new_history: corrected predictions
        :return:
        '''
        count = 0
        sum = 0
        for (x, y), old_pred in old_history.items():
            new_pred = new_history[(x, y)]

            if (old_pred > 0 and new_pred < 0) or (old_pred < 0 and new_pred > 0):
                count+=1
                dist = abs(old_pred - new_pred)
                sum += dist
        if count > 0 :
            print("avg of dist =", sum / count)
        return count

    def bulid_train_data(self, x1, y1, x2, y2, history, ground_truth, ):
        '''
        生成训练Slide Filter的训练样本集
        :param x1: 检测区域的左上角x
        :param y1: 检测区域的左上角y
        :param x2: 检测区域的右下角x
        :param y2: 检测区域的右下角y
        :param history: 预测的结果
        :param ground_truth: mask image
        :return:
        '''
        valid_area_width = x2 - x1
        valid_area_height = y2 - y1
        cancer_map = np.zeros((valid_area_height, valid_area_width),dtype=np.float)
        for (x, y), f in history.items():
            cancer_map[y, x] = f

        mask = ground_truth[y1:y2, x1:x2]
        X_data = []
        Y_data = []
        half_size = self.image_size // 2
        W = 1 # W = 4
        for x, y in history.keys():
            sx1 = x - half_size
            sx2 = x + half_size
            sy1 = y - half_size
            sy2 = y + half_size
            if sx1 < 0 or sx2 > valid_area_width or sy1 < 0 or sy2 > valid_area_height:
                continue
            sub_m = cancer_map[sy1:sy2, sx1:sx2]
            sub_y = np.any(mask[y - W : y + W, x - W: x + W])
            sparse_v = coo_matrix(sub_m)

            X_data.append(sparse_v)
            Y_data.append(int(sub_y))

        return X_data, Y_data

    def read_train_data(self, slice_dirname, chosen, np_ratio = 1.0, p_ratio = 1.0):
        '''
        读取预测结果，并生成训练样本
        :param slice_dirname:
        :param chosen: slide id的列表
        :param np_ratio: negative与positive样本的比例
        :param p_ratio: 包含positive样本占全部正样本的比例（有可能数据大于2GB不能存为npz文件，所以进行一些缩减）
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)
        mask_path = "{}/data/true_masks".format(self._params.PROJECT_ROOT)

        K = len("_history.npz")

        X = []
        Y = []
        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz" and "_history.npz" in result_file and "Test" not in result_file:
                print("loading data : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file))
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()

                mask_img = np.load("{}/{}_true_mask.npy".format(mask_path, slice_id))

                X_data, Y_data = self.bulid_train_data(x1, y1, x2, y2, history, mask_img)

                X.extend(X_data)
                Y.extend(Y_data)

        count = np.sum(Y)
        print("count of Y =", count, "len =", len(Y))

        # p_ratio = 0.6
        n_ratio = p_ratio * np_ratio * float(count) / len(Y)
        new_X = []
        new_Y = []

        for x, y in zip(X, Y):
            rand = random.random()
            if y:
                if rand <= p_ratio:
                    new_X.append(x)
                    new_Y.append(y)
            else:
                if rand <= n_ratio:
                    new_X.append(x)
                    new_Y.append(y)

        count = np.sum(new_Y)
        print("count of new Y =", count, "len =", len(new_Y))

        filename = "{}/data/slide_train_data{}_p{:.1f}_np{:.1f}.npz".format(self._params.PROJECT_ROOT, self.image_size,
                                                                         p_ratio, np_ratio)
        np.savez_compressed(filename, x=new_X, y=new_Y,)
        return

    def update_history(self, chosen, batch_size):
        '''
        使用Slide filter进行微调
        :param chosen: slide id列表
        :param batch_size:
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results".format(project_root)

        K = len("_history.npz")

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz" and "_history.npz" in result_file:
                print("loading data : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file), allow_pickle=True)
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()
                history2 = self.predict(x1, y1, x2, y2, history, batch_size=batch_size)

                history3 = self.fine_tuning(history, history2)
                modified_count = self.counting_history_changes(history, history3)
                print("{}, count of updated points = {}".format(slice_id, modified_count), )

                save_filename = "{}/results/{}_history_v{}.npz".format(self._params.PROJECT_ROOT, slice_id,self.image_size)
                np.savez_compressed(save_filename, x1=x1, y1=y1, x2=x2, y2=y2, scale=coordinate_scale, history=history3)
                print(">>> >>> ", save_filename, " saved!")

    def fine_tuning(self, old_history, new_history):
        '''
        对预测 特征进行加权平均
        :param old_history: 直接预测的结果
        :param new_history: 滤波器处理后的结果
        :return: 在概率上加权平均后的结果
        '''
        result = {}
        for (x, y), old_pred in old_history.items():
            new_pred = new_history[(x, y)]
            new_prob = 1 / (1 + np.exp(-new_pred))
            old_prob = 1 / (1 + np.exp(-old_pred))
            avg_prob = 0.5 * (old_prob + new_prob)
            result[(x, y)] = - np.log(1/avg_prob - 1)
        return result

    def summary_history(self, dir):
        '''
        统计 预测结果信息
        :param dir: 结果文件所在路径
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results/{}".format(project_root, dir)

        K = len("_history.npz")
        result = {}
        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]

            if ext_name == ".npz" and "_history.npz" in result_file:
                # print("loading data : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file), allow_pickle=True)
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()

                count = len(history.keys())
                area = (y2 - y1) * (x2 - x1)
                print("{}\t{}\t{}".format(slice_id, area, count))


class Sparse_Image_Dataset(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img = self.x[index].toarray()
        if self.y is not None:
            label = int(self.y[index])

        img_tensor = self.transform(img).type(torch.FloatTensor)
        if self.y is None:
            return img_tensor
        else:
            return img_tensor, label

    def __len__(self):
        return len(self.x)







