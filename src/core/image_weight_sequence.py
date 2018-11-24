#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-24'

"""
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from preparation.normalization import ImageNormalization

class ImageWeightSequence(Sequence):

    def __init__(self, x_set, y_set, w_set, batch_size, num_classes = 2, augmentation = False):
        self.x, self.y, self.w = x_set, y_set, w_set
        self.batch_size = batch_size
        self._augmentation = augmentation
        self.num_classes = num_classes

        # self.datagen = ImageDataGenerator(
        #     rotation_range=90,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     zoom_range=0.2,
        #     horizontal_flip=True,
        #     vertical_flip=True)
        #     # preprocessing_function=ImageNormalization.normalize_mean)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_w = self.w[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        return np.array([
            ImageNormalization.normalize_mean(imread(file_name))
            for file_name in batch_x]), to_categorical(batch_y, self.num_classes), np.array(batch_w)