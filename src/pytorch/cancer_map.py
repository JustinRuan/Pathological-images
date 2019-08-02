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
from skimage import morphology
from skimage.morphology import square
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from core.util import latest_checkpoint
from pytorch.loss_function import CenterLoss, LGMLoss

class CancerMapBuilder(object):
    def __init__(self, params, x1, y1, x2, y2, scale):
        self._params = params
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)
        self.valid_area_width = xx2 - xx1
        self.valid_area_height = yy2 - yy1

    def generating_probability_map(self, history, extract_scale, patch_size):
        seeds_scale = self._params.GLOBAL_SCALE
        amplify = extract_scale / seeds_scale
        selem_size = int(0.5 * patch_size / amplify)

        value = np.array(list(history.values()))
        point = list(history.keys())
        value_softmax = 1 / (1 + np.exp(-value))

        # 生成坐标网格
        grid_y, grid_x = np.mgrid[0: self.valid_area_height: 1, 0: self.valid_area_width: 1]
        cancer_map = griddata(point, value_softmax, (grid_x, grid_y), method='linear', fill_value=0)

        cancer_map = morphology.closing(cancer_map, square(2 * selem_size))
        cancer_map = morphology.dilation(cancer_map, square(selem_size))

        return cancer_map

    # def calc_probability_threshold(self, history):
    #     value = np.array(list(history.values()))
    #     # value_softmax = 1 / (1 + np.exp(-value))
    #     value = np.reshape(value, (-1, 1))
    #
    #     clustering = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100, compute_labels=True,
    #                                  batch_size=200, tol=1e-4).fit(value)
    #     cluster_centers = clustering.cluster_centers_.ravel()
    #     index = np.argmax(cluster_centers)
    #
    #     right_part = value[index == clustering.labels_.ravel()]
    #     if cluster_centers[index] > 0:
    #         low_thresh = np.min(right_part)
    #     else:
    #         ift = IsolationForest(behaviour='new', max_samples='auto', contamination=0.001)
    #         y = ift.fit_predict(right_part)
    #         outliers = right_part[y == -1]
    #
    #         low_thresh = np.min(outliers)
    #
    #     return 1 / (1 + np.exp(-low_thresh))


class Slide_CNN(nn.Module):

    def __init__(self, ):
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
        self.gap = nn.AdaptiveAvgPool2d((2, 2))
        feature_dim = 64 * 2 * 2
        self.dense = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图成 (batch_size, .....)
        x = self.dense(x)

        output = self.classifier(x)
        return output

class SlideClassifier(object):
    def __init__(self, params, model_name, patch_type):
        self._params = params

        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS
        self.image_size = int(patch_type)

        self.model_root = "{}/models/pytorch/Slide_{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
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
        return Slide_CNN()


    def train(self, batch_size, loss_weight, epochs):
        filename = "{}/data/slide_train_data.npz".format(self._params.PROJECT_ROOT)
        D = np.load(filename)
        X = D['x']
        Y = D['y']
        X = np.reshape(X, (-1,1,self.image_size, self.image_size))

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, )

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                    torch.from_numpy(y_train).long())
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(),
                                                   torch.from_numpy(y_test).long())
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        summary(model, torch.zeros((1, 1, self.image_size, self.image_size)), show_input=True)

        classifi_loss = nn.CrossEntropyLoss(weight=None)
        center_loss = CenterLoss(self.num_classes, 2)
        if self.use_GPU:
            model.to(self.device)
            classifi_loss.to(self.device)
            center_loss.to(self.device)

        # optimzer4nn
        # classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        classifi_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.001)
        # classifi_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, )
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


    def predict(self, x1, y1, x2, y2, history, batch_size):
        valid_area_width = x2 - x1
        valid_area_height = y2 - y1
        cancer_map = np.zeros((valid_area_height, valid_area_width),dtype=np.float)
        for (x, y), f in history.items():
            cancer_map[y, x] = f

        X_data = []
        half_size = 64
        xy = []
        prob = []
        for (x, y), f in history.items():
            sx1 = x - half_size
            sx2 = x + half_size
            sy1 = y - half_size
            sy2 = y + half_size
            if sx1 < 0 or sx2 > valid_area_width or sy1 < 0 or sy2 > valid_area_height:
                continue
            sub_m = cancer_map[sy1:sy2, sx1:sx2]
            X_data.append(sub_m)
            xy.append((x, y))
            prob.append(f)

        if self.model is None:
            self.model = self.load_model(model_file=None)

            # 关闭求导，节约大量的显存
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.to(self.device)
            self.model.eval()

        X_data = np.array(X_data).reshape((-1,1,self.image_size,self.image_size))
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_data).float(),)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        data_len = len(test_loader)

        prediction = []
        for step, xx in enumerate(test_loader):
            b_x = Variable(xx[0].to(self.device))

            output = self.model(b_x)  # model最后不包括一个softmax层

            prediction.extend(output.cpu().numpy())
            print('predicting => %d / %d ' % (step + 1, data_len))

        count = 0
        for (x, y), f, pred, in zip(xy, prob, prediction):
            history[(x, y)] = pred[1]

            if pred[1]*f < 0:
                # print(f, pred[1])
                count+=1

        print("count of updated points, ", count)
        return history


    def bulid_train_data(self, x1, y1, x2, y2, history, ground_truth, ):
        valid_area_width = x2 - x1
        valid_area_height = y2 - y1
        cancer_map = np.zeros((valid_area_height, valid_area_width),dtype=np.float)
        for (x, y), f in history.items():
            cancer_map[y, x] = f

        mask = ground_truth[y1:y2, x1:x2]
        X_data = []
        Y_data = []
        half_size = 64

        for x, y in history.keys():
            sx1 = x - half_size
            sx2 = x + half_size
            sy1 = y - half_size
            sy2 = y + half_size
            if sx1 < 0 or sx2 > valid_area_width or sy1 < 0 or sy2 > valid_area_height:
                continue
            sub_m = cancer_map[sy1:sy2, sx1:sx2]
            sub_y = np.any(mask[y - 4 : y + 4, x - 4: x + 4])
            X_data.append(sub_m)
            Y_data.append(int(sub_y))

        return X_data, Y_data

    def read_train_data(self, slice_dirname, chosen):
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

            if ext_name == ".npz":
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

        ratio = float(count) / len(Y)
        new_X = []
        new_Y = []

        for x, y in zip(X, Y):
            if y:
                new_X.append(x)
                new_Y.append(y)
            else:
                rand = random.random()
                if rand <= ratio:
                    new_X.append(x)
                    new_Y.append(y)

        count = np.sum(new_Y)
        print("count of new Y =", count, "len =", len(new_Y))

        filename = "{}/data/slide_train_data.npz".format(self._params.PROJECT_ROOT)
        np.savez_compressed(filename, x=new_X, y=new_Y,)
        return











