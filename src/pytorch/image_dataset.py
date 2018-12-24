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