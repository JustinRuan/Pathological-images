#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-06'

"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #CPU

import math
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
from .net.densenet import create_densenet_40, create_densenet_22


class CNN_Classifier(object):

    def __init__(self, params, model_name, patch_type):

        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        if self.patch_type == "500_128":
            self.num_classes = 2
            self.input_shape = (128, 128, 3)
        elif self.patch_type in ["2000_256", "4000_256"]:
            self.num_classes = 2
            self.input_shape = (256, 256, 3)
        elif self.patch_type == "cifar10":
            self.num_classes = 10
            self.input_shape = (32, 32, 3)
        elif self.patch_type == "cifar100":
            self.num_classes = 100
            self.input_shape = (32, 32, 3)

        self.model_root = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)

    def create_initial_model(self):

        if self.model_name == "simple_cnn":
            model = create_simple_cnn(num_classes=self.num_classes, input_shape=self.input_shape, top_units=256)

        elif self.model_name=="densenet_40":
            model = create_densenet_40(nb_classes=self.num_classes, input_shape=self.input_shape)

        elif self.model_name == "densenet_22":
            model = create_densenet_22(nb_classes=self.num_classes, input_shape=self.input_shape)

        return model

    def load_model(self, model_file = None):

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            model = load_model(model_file)
            return model
        else:
            checkpoint_dir = self.model_root
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = util.latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                model = load_model(latest)
            else:
                model = self.create_initial_model()
            return model

    # def generate_arrays_on_batch(self, x_set, y_set, batch_size, num_classes):
    #     len_data = math.ceil(len(x_set) / batch_size)
    #
    #     while 1:
    #         for idx in range(len_data):
    #             batch_x = x_set[idx * batch_size:(idx + 1) * batch_size]
    #             batch_y = y_set[idx * batch_size:(idx + 1) * batch_size]
    #             yield (np.array(batch_x), to_categorical(batch_y, num_classes))

    def train_model_cifar(self, batch_size = 100, epochs = 20, initial_epoch = 0):
        if self.patch_type == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif self.patch_type == "cifar100":
            (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        else:
            return

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

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
        optimizer = RMSprop(lr=1e-3, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose=1,
                      # callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      callbacks=[cp_callback, early_stopping, reduce_lr],
                      validation_data=(x_test, y_test),initial_epoch=initial_epoch)

    def train_model(self, samples_name, batch_size, augmentation, epochs, initial_epoch):
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation)

        checkpoint_path = self.model_root + "/cp-{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5"

        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False,
            # Save weights, every 5-epochs.
            period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                          epsilon=0.0001, cooldown=0, min_lr=0)

        model = self.load_model()
        print(model.summary())
        # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = RMSprop(lr=1e-3, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        steps_per_epoch = min(800, train_gen.__len__())
        validation_steps = min(500, test_gen.__len__())

        model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, workers=self.NUM_WORKERS,
                            # callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            callbacks=[cp_callback, early_stopping, reduce_lr],
                            validation_data=test_gen, validation_steps=validation_steps, initial_epoch=initial_epoch)
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
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size, self.num_classes, augmentation[0])

        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size, self.num_classes, augmentation[1])
        return  train_gen, test_gen

    def predict(self, src_img, scale, patch_size, seeds, model_file):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.load_model(model_file)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        result = []
        for x, y in seeds:
            block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(ImageNormalization.normalize_mean(img))
            x = np.expand_dims(x, axis=0)

            predictions = model.predict(x)
            class_id = np.argmax(predictions[0])
            probability = predictions[0][class_id]
            result.append((class_id, probability))
        return result

    def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size, model_file):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.load_model(model_file)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        image_itor = SeedSequence(src_img, scale, patch_size, seeds, batch_size)

        predictions = model.predict_generator(image_itor, verbose=1, workers=self.NUM_WORKERS)
        result = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            result.append((class_id, probability))

        return result

    def predict_test_file(self, model_file, test_file_list, batch_size):
        model = self.load_model(model_file)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        Xtest = []
        Ytest = []

        for item in test_file_list:
            test_list = "{}/{}".format(self._params.PATCHS_ROOT_PATH, item)
            Xtest1, Ytest1 = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)

            Xtest.extend(Xtest1)
            Ytest.extend(Ytest1)

        image_itor = ImageSequence(Xtest, Ytest, batch_size, self.num_classes)

        predictions = model.predict_generator(image_itor, verbose=1, workers=self.NUM_WORKERS)
        predicted_tags = []
        predicted_probability = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            predicted_tags.append(class_id)
            predicted_probability.append(probability)

        print("Classification report for classifier:\n%s\n"
              % ( metrics.classification_report(Ytest, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest, predicted_tags))

        print("average predicted probability = %s" % np.mean(predicted_probability))

        Ytest = np.array(Ytest)
        predicted_tags = np.array(predicted_tags)
        predicted_probability = np.array(predicted_probability)
        TP = np.logical_and(Ytest == 1, predicted_tags == 1)
        FP = np.logical_and(Ytest == 0, predicted_tags == 1)
        TN = np.logical_and(Ytest == 0, predicted_tags == 0)
        FN = np.logical_and(Ytest == 1, predicted_tags == 0)

        print("average TP probability = %s" % np.mean(predicted_probability[TP]))
        print("average FP probability = %s" % np.mean(predicted_probability[FP]))
        print("average TN probability = %s" % np.mean(predicted_probability[TN]))
        print("average FN probability = %s" % np.mean(predicted_probability[FN]))
        return

