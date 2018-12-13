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
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        file_name = self.x[index]
        label = self.y[index]
        img = ImageNormalization.normalize_mean(imread(file_name)) / 255
        img = self.transform(img).type(torch.FloatTensor)
        return img, label

    def __len__(self):
        return len(self.x)
