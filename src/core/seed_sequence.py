#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-10'

"""

from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from preparation.normalization import ImageNormalization

# Here, `x_set` is list of seeds to the images
class SeedSequence(Sequence):

    def __init__(self, src_img, scale, patch_size, output_size, seeds, batch_size):
        self.x = seeds
        self.batch_size = batch_size
        self.src_img = src_img
        self.scale = scale
        self.patch_size = patch_size
        self.output_size = output_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        img_list = []
        for x, y in batch_x:
            block = self.src_img.get_image_block(self.scale, x, y, self.patch_size, self.patch_size)
            img = block.get_img()
            img_list.append(img)

        return np.array([
                resize(img, (self.output_size, self.output_size, 3))
                for img in img_list])