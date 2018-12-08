#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-06'

"""
import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential, load_model

# def create_simple_cnn(num_classes, input_shape, top_units):
#
#     model = Sequential()
#     # input: 128x128 images with 3 channels -> (128, 128, 3) tensors.
#     # this applies 16 convolution filters of size 3x3 each.
#     # 第一个卷积层, 128x128  => 64x64
#     model.add(Conv2D(16, kernel_size=(3, 3), strides=2, padding="same", activation='relu',
#                      input_shape=input_shape, kernel_initializer='random_uniform')) # (128, 128, 3)
#     # 池化层 64 x 64 => 32 x 32
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#     # 第二个卷积层 32 x 32 => 32 x 32
#     model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer='random_uniform'))
#     # 池化层 32 x 32 => 16 x 16
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#     # 第三个卷积层 16 x 16 => 16 x 16
#     model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer='random_uniform'))
#     # 池化层 16 x 16 => 8 x 8
#     model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#
#     model.add(Flatten())
#     # 全连接层
#     model.add(Dense(top_units, activation='relu')) # 512
#     model.add(Dense(num_classes, activation='softmax'))
#
#     return model

# cifar 10的准确率为76%
def create_simple_cnn(num_classes, input_shape, top_units):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                     input_shape=input_shape, kernel_initializer='random_uniform')) # (128, 128, 3)
    model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Conv2D(48, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(GlobalAveragePooling2D())

    # 全连接层
    model.add(Dense(top_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model