#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

# from tensorflow.keras.utils import Sequence, to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from preparation.normalization import ImageNormalization

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class ImageSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, output_size, num_classes = 2, augmentation = False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self._augmentation = augmentation
        self.num_classes = num_classes
        self.output_size = output_size

        self.datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
            # preprocessing_function=ImageNormalization.normalize_mean)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        # if self._augmentation:
        #     return np.array([
        #         # resize(imread(file_name),(299,299))
        #         resize(ImageNormalization.normalize_mean(self.datagen.random_transform(imread(file_name))),
        #                (self.output_size, self.output_size))
        #         for file_name in batch_x]), to_categorical(batch_y, self.num_classes)
        # else:
        #     return np.array([
        #         resize(ImageNormalization.normalize_mean(imread(file_name)),(self.output_size, self.output_size))
        #         for file_name in batch_x]), to_categorical(batch_y, self.num_classes)

        # resize 太费时间，暂时先不用，提高程序调试的效率
        return np.array([
                ImageNormalization.normalize_mean(imread(file_name))
                for file_name in batch_x]), to_categorical(batch_y, self.num_classes)