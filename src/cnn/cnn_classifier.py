#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-06'

"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #CPU

import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn import metrics
from skimage import io, util
from core import *
from core.util import read_csv_file
from preparation.normalization import ImageNormalization

from keras.datasets import cifar10
from keras.datasets import cifar100

from .net.simple_cnn import create_simple_cnn


class CNN_Classifier(object):

    def __init__(self, params, model_name, patch_type):

        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

    def create_initial_model(self):

        if self.model_name == "simple_cnn":
            if self.patch_type == "500_128":
                model = create_simple_cnn(num_classes = 2, input_shape = (128, 128, 3), top_units = 512)
            elif self.patch_type in ["2000_256", "4000_256"] :
                model = create_simple_cnn(num_classes=2, input_shape=(256, 256, 3), top_units=512)
            elif self.patch_type == "cifar10":
                model = create_simple_cnn(num_classes=10, input_shape=(32, 32, 3), top_units=512)
            elif self.patch_type == "cifar100":
                model = create_simple_cnn(num_classes=100, input_shape=(32, 32, 3), top_units=512)

        return model

    def load_model(self, model_file = None):

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            model = load_model(model_file)
            return model
        else:
            checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = util.latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                model = load_model(latest)
            else:
                model = self.create_initial_model()
            return model


    def train_model_cifar(self, batch_size = 100, epochs = 20, initial_epoch = 0):
        if self.patch_type == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)
        elif self.patch_type == "cifar100":
            (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
            y_train = to_categorical(y_train, 100)
            y_test = to_categorical(y_test, 100)
        else:
            return



        checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5"

        model = self.load_model(model_file=None)

        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False,
            # Save weights, every 5-epochs.
            period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                      mode='auto',
                                                      epsilon=0.0001, cooldown=0, min_lr=0)

        print(model.summary())
        # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose=1,
                      # callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      callbacks=[cp_callback, early_stopping, reduce_lr],
                      validation_data=(x_test, y_test),initial_epoch=initial_epoch)
