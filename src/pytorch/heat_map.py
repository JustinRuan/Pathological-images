#!/usr/bin/env python
# encoding: utf-8
'''
@author: Justin Ruan
@license: 
@contact: ruanjun@whut.edu.cn
@time: 2020-01-10
@desc:
'''

import os, datetime
import numpy as np
from scipy import ndimage as nd
from skimage import measure
from skimage.transform import resize
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

from core.util import latest_checkpoint, is_tumor_by_code
from pytorch.loss_function import CenterLoss, LGMLoss, LGMLoss_v0


class Slide_Image_Dataset(Dataset):
    def __init__(self, filename_set):
        self.filenames = filename_set
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        filepath = self.filenames[index]
        result = np.load(filepath, allow_pickle=True)
        img = result["x"]
        label = result["y"]

        img_tensor = self.transform(img).type(torch.FloatTensor)
        label_tensor = self.transform(label).type(torch.FloatTensor)

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.filenames)


class Slide_FCN(nn.Module):

    def __init__(self, ):
        super(Slide_FCN, self).__init__()

        # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # m x m => m x m
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)  # m x m => m/2 x m/2
        # self.conv4 = nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0)  # m/2 x m/2 => m/4 x m/4

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),  # m x m => m/2 x m/2
            nn.ReLU(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # m/2 x m/2 => m/4 x m/4
            nn.ReLU(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # m/4 x m/4 => m/4 x m/4
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.conv3(x)

        return y


class HeatMapBuilder(object):
    def __init__(self, params, model_name):
        self._params = params

        self.model_name = model_name
        self.data_path = "{}//data//slide".format(self._params.PROJECT_ROOT)

        self.NUM_WORKERS = params.NUM_WORKERS

        self.model_root = "{}/models/pytorch/{}".format(self._params.PROJECT_ROOT, self.model_name)
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
        return Slide_FCN()

    def bulid_train_data(self, slide_id, x1, y1, x2, y2, history, ground_truth, level):
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
        cancer_map = np.zeros((valid_area_height, valid_area_width), dtype=np.float)
        for (x, y), f in history.items():
            cancer_map[y, x] = f

        L0_RESOLUTION = 0.243  # pixel resolution at level 0
        r, c = valid_area_height, valid_area_width
        m = np.power(2, level - 5)
        if ground_truth is not None:
            mask = ground_truth[y1:y2, x1:x2]

            if level > 5:
                pixelarray = resize(mask, (r // m, c // m))
            else:
                pixelarray = mask
            Threshold = 75 / (L0_RESOLUTION * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
            distance = nd.distance_transform_edt(255 * (1 - pixelarray))
            binary = distance < Threshold
            filled_image = nd.morphology.binary_fill_holes(binary)

            mask = measure.label(filled_image, connectivity=2)

        else:
            mask = np.zeros((r // m, c // m), dtype=np.bool)

        filename = "{}/slide_{}.npz".format(self.data_path, slide_id)
        np.savez_compressed(filename, x=cancer_map, y=mask, )

        return

    def create_train_data(self, chosen,):
        '''
        读取预测结果，并生成训练样本
        :param slice_dirname:
        :param chosen: slide id的列表
        :param np_ratio: negative与positive样本的比例
        :param p_ratio: 包含positive样本占全部正样本的比例（有可能数据大于2GB不能存为npz文件，所以进行一些缩减）
        :return:
        '''
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/results/history".format(project_root)
        mask_path = "{}/data/true_masks".format(self._params.PROJECT_ROOT)

        K = len("_history.npz")

        for result_file in os.listdir(save_path):
            ext_name = os.path.splitext(result_file)[1]
            slice_id = result_file[:-K]
            if chosen is not None and slice_id not in chosen:
                continue

            if ext_name == ".npz" and "_history.npz" in result_file and "Test" not in result_file:
                print("loading data : {}".format(slice_id))
                result = np.load("{}/{}".format(save_path, result_file), allow_pickle=True)
                x1 = result["x1"]
                y1 = result["y1"]
                x2 = result["x2"]
                y2 = result["y2"]
                coordinate_scale = result["scale"]
                assert coordinate_scale == 1.25, "Scale is Error!"

                history = result["history"].item()

                is_tumor = is_tumor_by_code(slice_id)
                if is_tumor:
                    result = np.load("{}/{}_true_mask.npz".format(mask_path, slice_id), allow_pickle=True)
                    mask_img = result["mask"]
                else:
                    mask_img = None

                self.bulid_train_data(slice_id, x1, y1, x2, y2, history, mask_img, level=7)

        return

    def loading_data_for_train(self):
        project_root = self._params.PROJECT_ROOT
        save_path = "{}/data/slide".format(project_root)

        filenames = []
        for save_file in os.listdir(save_path):
            ext_name = os.path.splitext(save_file)[1]
            if ext_name == ".npz":
                filenames.append("{}/{}".format(save_path, save_file))

        d_train, d_test = train_test_split(filenames, test_size=0.05, )
        train_data = Slide_Image_Dataset(d_train)
        test_data = Slide_Image_Dataset(d_test)

        return train_data, test_data

    def train(self, batch_size, epochs):

        train_data, test_data = self.loading_data_for_train()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=self.NUM_WORKERS)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=self.NUM_WORKERS)

        model = self.load_model(model_file=None)
        summary(model, torch.zeros((1, 1, 1000, 1000)), show_input=False)

        criterion = nn.BCELoss(reduction='mean')
        if self.use_GPU:
            model.to(self.device)
            criterion.to(self.device)

        # optimzer4nn
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4) #学习率为0.01的学习器
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, )
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
                loss = criterion(output, b_y)

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                # 数据统计
                running_loss = loss.item()
                total_loss += running_loss

                if step % 2 == 0:
                    endtime = datetime.datetime.now()
                    remaining_time = (train_data_len - step)* (endtime - starttime).seconds / (step + 1)
                    print('%d / %d ==> Loss: %.4f \t total loss %.4f: ,  remaining time: %d (s)'
                          % (step, train_data_len, running_loss, total_loss, remaining_time))

            scheduler.step(total_loss)

            running_loss=0.0
            running_corrects=0
            model.eval()
            # 开始评估
            for x, y in test_loader:
                b_x = Variable(x.to(self.device))
                b_y = Variable(y.to(self.device))

                output = model(b_x)
                loss = criterion(output, b_y)

                running_loss += loss.item() * b_x.size(0)

            test_data_len = test_data.__len__()
            epoch_loss=running_loss / test_data_len

            save_filename = self.model_root + "/{}_cp-{:04d}-{:.4f}-{:.4f}.pth".format(
                self.model_name, epoch + 1, epoch_loss, epoch_loss)
            torch.save(model.state_dict(), save_filename)
            print("Saved ", save_filename)
        return