#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-10'

"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #CPU

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from skimage import io, util
from core import *
from core.util import read_csv_file

class cnn_simple_5x128(object):

    def __init__(self, params, model_name):
        '''
         初始化CNN分类器
        :param params: 参数
        :param model_name: 使用的模型文件
        :param samples_name: 使用的标本集的关键字（标记符），为None时是进入预测模式
        '''

        model_name = model_name + "_keras"
        self._params = params
        self.model_name = model_name

        self.model_root = "{}/models/{}".format(self._params.PROJECT_ROOT, model_name)
        return

    def create_model(self):
        num_classes = 2

        model = Sequential()
        # input: 128x128 images with 3 channels -> (128, 128, 3) tensors.
        # this applies 16 convolution filters of size 3x3 each.
        # 第一个卷积层, 128x128  => 64x64
        model.add(Conv2D(16, kernel_size = (3, 3), strides= 2, padding="same", activation='relu',
                         input_shape=(128, 128, 3)))
        # 池化层 64 x 64 => 32 x 32
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
        # 第二个卷积层 32 x 32 => 32 x 32
        model.add(Conv2D(8, kernel_size = (3, 3), strides= 1, padding="same", activation='relu'))
        # 池化层 32 x 32 => 16 x 16
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
        # 第三个卷积层 16 x 16 => 16 x 16
        model.add(Conv2D(1, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
        # 池化层 16 x 16 => 8 x 8
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

        model.add(Flatten())
        # 全连接层
        # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        # # model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        # model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, self.model_name)
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if not latest is None:
            print("loading >>> ", latest, " ...")
            model.load_weights(latest)

        return model

    def train_model(self, samples_name, batch_size, augmentation):
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation)

        # checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, samples_name)
        checkpoint_dir = self.model_root
        checkpoint_path = self.model_root + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        model = self.create_model()
        # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit_generator(train_gen, steps_per_epoch=20, epochs=20, verbose=1,
                            callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            validation_data=test_gen, validation_steps=6)
        return

    def load_data(self, samples_name, batch_size, augmentation = (False, False)):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :param batch_size: 图片读取时的每批的图片数量
        :return:用于train和test的两个Sequence
        '''
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size, augmentation[0])

        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size, augmentation[1])
        return  train_gen, test_gen

    def predict(self):

        return

