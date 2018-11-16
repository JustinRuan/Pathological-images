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
from preparation.normalization import ImageNormalization

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

    def create_model(self, model_file = None):
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
        model.add(Conv2D(32, kernel_size = (3, 3), strides= 1, padding="same", activation='relu'))
        # 池化层 32 x 32 => 16 x 16
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
        # 第三个卷积层 16 x 16 => 16 x 16
        model.add(Conv2D(4, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
        # 池化层 16 x 16 => 8 x 8
        model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

        model.add(Flatten())
        # 全连接层
        # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        # # model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        # model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        if model_file is None:
            checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, self.model_name)
            latest = tf.train.latest_checkpoint(checkpoint_dir)

            if not latest is None:
                print("loading >>> ", latest, " ...")
                model.load_weights(latest)
        else:
            model_path = "{}/models/trained/{}".format(self._params.PROJECT_ROOT, model_file)
            print("loading >>> ", model_path, " ...")
            model.load_weights(model_path)

        return model

    def train_model(self, samples_name, batch_size, augmentation):
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation)

        train_len = train_gen.__len__()
        test_len = test_gen.__len__()

        # checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, samples_name)
        checkpoint_dir = self.model_root
        checkpoint_path = self.model_root + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        model = self.create_model()
        print(model.summary())
        # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit_generator(train_gen, steps_per_epoch=train_len, epochs=20, verbose=1,
                            callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            validation_data=test_gen, validation_steps=test_len, initial_epoch = 0)
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

    def predict(self, src_img, scale, patch_size, seeds):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.create_model()
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

    def predict_on_batch(self, src_img, scale, patch_size, seeds, batch_size):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.create_model()
        optimizer = RMSprop(lr=1e-4, rho=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        image_itor = SeedSequence(src_img, scale, patch_size, seeds, batch_size)

        predictions = model.predict_generator(image_itor, verbose=1)
        result = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            result.append((class_id, probability))

        return result

    def predict_test_file(self, model_file, test_file_list):
        model = self.create_model(model_file)
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

        image_itor = ImageSequence(Xtest, Ytest, 20)

        predictions = model.predict_generator(image_itor, verbose=1)
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

    # 没有搞定这种写法
    # def get_patches_itor(self, src_img, scale, patch_size, seeds, batch_size):
    #     '''
    #     将所有的图块中心点集，分割成更小的集合，以便减小图块载入的内存压力
    #     :param seeds: 所有的需要计算的图块中心点集
    #     :param batch_size: 每次读入到内存的图块数量（不是Tensor的批处理数量）
    #     :return: 种子点的迭代器
    #     '''
    #     len_seeds = len(seeds)
    #     start_id = 0
    #     for end_id in range(batch_size + 1, len_seeds, batch_size):
    #         result = []
    #         for x, y in seeds[start_id : end_id]:
    #             block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
    #             img = block.get_img()
    #
    #             # x = image.img_to_array(ImageNormalization.normalize_mean(img))
    #             # x = np.expand_dims(x, axis=0)
    #             result.append(np.array(ImageNormalization.normalize_mean(img)))
    #
    #         start_id = end_id
    #         yield result
    #
    #     result = []
    #     if start_id < len_seeds:
    #         for x, y in seeds[start_id: len_seeds]:
    #             block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
    #             img = block.get_img()
    #
    #             # x = image.img_to_array(ImageNormalization.normalize_mean(img))
    #             # x = np.expand_dims(x, axis=0)
    #             result.append(np.array(ImageNormalization.normalize_mean(img)))
    #         yield result
    #     return