#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from core.util import read_csv_file
from skimage.io import imread
from preparation.normalization import ImageNormalization
from skimage.transform import resize



class Image_Dataset(Dataset):
    def __init__(self, x_set, y_set, transform = None):
        self.x, self.y = x_set, y_set
        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
            # self.transform = torchvision.transforms.Compose([
            #     torchvision.transforms.ToTensor(),
            #     # RGB道通的归一化
            #     torchvision.transforms.Normalize((0.7886, 0.6670, 0.7755), (0.1324, 0.1474, 0.1172)),
            #     # R,G,B每层的归一化用到的均值和方差
            # ])
        else:
            self.transform = transform

        if True:
            # 72.6668955191 16.8991760939 -9.97997070951 13.427538741 5.76732834018 4.89131967314
            self.normal = ImageNormalization(None, avg_mean_l=72.66, avg_mean_a=16.9, avg_mean_b=-9.98,
                                             avg_std_l=13.43, avg_std_a=5.767, avg_std_b=4.891)
        else:
            self.normal = None

    def __getitem__(self, index):
        file_name = self.x[index]
        label = self.y[index]

        # #RGB归一化前除255
        # img = np.divide(imread(file_name), 255.0, dtype=np.float32)

        # LAB空间归一化
        img = imread(file_name)
        if self.normal is not None:
            img = self.normal.normalize(img)

        img = self.transform(img).type(torch.FloatTensor)
        return img, label

    def __len__(self):
        return len(self.x)

class Image_Dataset_MSC(Dataset):
    def __init__(self, x10_set, x20_set, x40_set, y_set, transform = None):
        self.y = y_set
        self.x10 = x10_set
        self.x20 = x20_set
        self.x40 = x40_set

        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        file_name10 = self.x10[index]
        file_name20 = self.x20[index]
        file_name40 = self.x40[index]
        label = self.y[index]

        # img10 = ImageNormalization.normalize_mean(imread(file_name10)) / 255
        # img20 = ImageNormalization.normalize_mean(imread(file_name20)) / 255
        # img40 = ImageNormalization.normalize_mean(imread(file_name40)) / 255
        img10 = imread(file_name10) / 255
        img20 = imread(file_name20) / 255
        img40 = imread(file_name40) / 255

        img = np.concatenate((img10, img20, img40), axis=-1)

        img = self.transform(img).type(torch.FloatTensor)

        return img, label

    def __len__(self):
        return len(self.y)


# class Image_Dataset2(Dataset):
#     def __init__(self, x_set, y_set, y2_set):
#         self.x, self.y ,self.y2 = x_set, y_set,y2_set
#         self.transform = torchvision.transforms.ToTensor()
#
#     def __getitem__(self, index):
#         file_name = self.x[index]
#         label1 = self.y[index]
#         label2 = self.y2[index]
#         # img = ImageNormalization.normalize_mean(imread(file_name)) / 255
#         img = imread(file_name) / 255
#         if img.shape==(128,128,3):
#             img=resize(img, (256,256,3))
#         img = self.transform(img).type(torch.FloatTensor)
#         return img, [label1, label2]
#
#     def __len__(self):
#         return len(self.x)

# Exception: ctypes objects containing pointers cannot be pickled
# class Seed_Dataset(Dataset):
#     def __init__(self, src_img, scale, patch_size, seeds):
#         self.seeds = seeds
#         # self.batch_size = batch_size
#         self.src_img = src_img
#         self.scale = scale
#         self.patch_size = patch_size
#         # self.output_size = output_size
#
#     def __getitem__(self, index):
#         x, y = self.seeds[index]
#         block = self.src_img.get_image_block(self.scale, x, y, self.patch_size, self.patch_size)
#         img = block.get_img() / 255
#         img = self.transform(img).type(torch.FloatTensor)
#         return img
#
#     def __len__(self):
#         return len(self.seeds)