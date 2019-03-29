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
        else:
            self.transform = transform

    def __getitem__(self, index):
        file_name = self.x[index]
        label = self.y[index]
        # img = ImageNormalization.normalize_mean(imread(file_name)) / 255
        img = imread(file_name) / 255
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