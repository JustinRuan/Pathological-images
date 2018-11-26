#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #CPU

import numpy as np
import tensorflow as tf
import keras
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121

from keras import backend as K
from keras.losses import categorical_crossentropy

from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout, Activation
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adamax, Nadam, Adadelta
from keras.preprocessing import image
from keras.utils import to_categorical, plot_model

from preparation.normalization import ImageNormalization
from core.util import read_csv_file
from core import *

NUM_CLASSES = 4
NUM_WORKERS = 1

# def categorical_crossentropy2(y_true, y_pred):
#   return K.categorical_crossentropy(y_true, y_pred)

class Transfer(object):
    # patch_type 500_128, 2000_256
    def __init__(self, params, model_name, patch_type):
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        return

    def extract_features(self,base_model, src_img, scale, patch_size, seeds):
        '''
        从切片中提取图块，并用网络提取特征
        :param src_img: 切片图像
        :param scale: 提取的倍镜数
        :param patch_size: 图块大小
        :param seeds: 图块中心点的坐标
        :return: 特征
        '''
        # create the base pre-trained model
        # base_model = InceptionV3(weights='imagenet', include_top=False, pooling = 'avg')
        # base_model = DenseNet121(weights='imagenet', include_top=False, pooling = 'avg')
        # print(base_model.summary())
        f_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        features = []
        for x, y in seeds:
            block= src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(ImageNormalization.normalize_mean(img))
            x = np.expand_dims(x, axis=0)
            # x = preprocess_input(x) //训练时没有使用预处理，这里也不能调用

            feature = f_model.predict(x)
            features.append(feature)

        return features

    def add_new_top_layers(self, input_layer, include_avg_pool = True):
        if include_avg_pool:
            x = GlobalAveragePooling2D(name='top_avg_pool')(input_layer)
        else:
            x = input_layer

        # x = BatchNormalization(name="top_bn")(x)
        # x = Dropout(0.5)(x)
        # let's add a fully-connected layer, kernel_regularizer=regularizers.l2(0.01),
        x = Dense(1024, activation='relu', name="top_Dense")(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)
        return predictions

    def create_initial_model(self):
        if self.model_name == "inception_v3":
            base_model = InceptionV3(weights='imagenet', include_top=False)
            predictions = self.add_new_top_layers(base_model.output)
            model = Model(inputs=base_model.input, outputs=predictions)
            return model

        elif self.model_name == "densenet121":

             return

    def load_model(self, mode, model_file = None):
        if mode == 0:  # "load_entire_model"
            checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = util.latest_checkpoint(checkpoint_dir)
            if latest is not None:
                model = load_model(latest, compile=False)
            else:
                if model_file is not None and os.path.exists(model_file):
                    model = load_model(model_file, compile=False)
                else:
                    model = self.create_initial_model()

            return model
        elif mode == 1:  # "extract features for transfer learning"

            full_model = self.create_initial_model()
            feature_model = Model(inputs=full_model.input, outputs=full_model.get_layer('top_avg_pool').output)
            return feature_model

        elif mode == 2:
            checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest_top = util.latest_checkpoint(checkpoint_dir)
            if latest_top is not None:
                top_model = load_model(latest_top, compile=False)
            else:
                if self.model_name == "inception_v3":
                    input_layer = Input(shape=(2048,))
                predictions = self.add_new_top_layers(input_layer, False)
                top_model = Model(inputs=input_layer, outputs=predictions)
            return top_model

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
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size, NUM_CLASSES, augmentation[0])
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size, NUM_CLASSES, augmentation[1])
        return  train_gen, test_gen


    def fine_tuning_model(self, freezed_num, optimizer, samples_name,batch_size, epochs = 20, initial_epoch = 0):
        '''
        微调模型，训练时冻结网络中0到freezed_num层
        :param model_dir:训练时checkpoint文件的存盘路径
        :param samples_name:训练所使用的图片列表文件的代号
        :param freezed_num: 训练时冻结网络中 0 到 freezed_num 层
        :param optimizer: 训练用，自适应学习率算法
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size)

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.h5"

        model = self.load_model(mode = 0)

        for i, layer in enumerate(model.layers[:freezed_num]):
            layer.trainable = False
            print( " freezed ", i, layer.name, sep="\t\t")
        for i, layer in enumerate(model.layers[freezed_num:]):
            layer.trainable = True
            print("trainable", i + freezed_num, layer.name, sep="\t\t")

        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False,
            # Save weights, every 5-epochs.
            period=3)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                          epsilon=0.0001, cooldown=0, min_lr=0)

        # compile the model (should be done *after* setting layers to non-trainable) 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())
        train_step = min(200, train_gen.__len__())
        test_step = min(100, test_gen.__len__())
        # train the model on the new data for a few epochs
        model.fit_generator(train_gen, steps_per_epoch=train_step, epochs=epochs, verbose=1, workers=NUM_WORKERS,
                            # callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            callbacks=[cp_callback, early_stopping, reduce_lr],
                            validation_data=test_gen, validation_steps=test_step, initial_epoch = initial_epoch)
        return


    def fine_tuning_inception_v3_249(self, samples_name, batch_size, epochs, initial_epoch):
        '''
        训练全连接层，和最后一部分的原网络
        :param samples_name:
        :return:
        '''
        # optimizer = SGD(lr=1e-4, momentum=0.9)
        # optimizer = RMSprop(lr=1e-4, rho=0.9)
        optimizer = SGD(lr=1e-3, momentum=0.9)
        # 最后两个Inception的位置：249, 最后一个的位置：280, Top的位置：311
        self.fine_tuning_model(249, optimizer, samples_name, batch_size, epochs, initial_epoch)

    def extract_features_for_train(self, samples_name, batch_size):
        '''
        从训练的图块中提取 特征， 并存盘
        :param samples_name: 图块文件所在列表txt文件
        :return: 生成两个特征文件
        '''
        train_list = "{}_train.txt".format(samples_name)
        test_list = "{}_test.txt".format(samples_name)
        model = self.load_model(mode=1)
        print(model.summary())
        self.extract_features_save_to_file(model, train_list, batch_size)
        self.extract_features_save_to_file(model, test_list, batch_size)
        return

    def extract_features_save_to_file(self, model, samples_file_path, batch_size, augmentation = False, aug_multiple = 1):
        sample_name = samples_file_path[:-4]
        file_list = "{}/{}".format(self._params.PATCHS_ROOT_PATH, samples_file_path)
        if not augmentation:
            save_path = "{}/data/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, sample_name)
        else:
            save_path = "{}/data/{}_{}_Aug{}".format(self._params.PROJECT_ROOT, self.model_name,
                                                            sample_name, aug_multiple)

        X, Y = read_csv_file(self._params.PATCHS_ROOT_PATH, file_list)
        data_gen = ImageSequence(X, Y, batch_size, augmentation)

        if not augmentation:
            aug_multiple = 1

        step_count = int(aug_multiple * len(Y) / batch_size)

        features = model.predict_generator(data_gen, steps=step_count, verbose=1,workers=NUM_WORKERS)
        labels = Y[:len(features)]
        np.savez(save_path + "_features", features, labels)
        return

    def fine_tuning_top_model_saved_file(self, train_filename, test_filename,
                                         batch_size = 100, epochs = 20, initial_epoch = 0):
        '''
        使用存盘的特征文件来训练 全连接层
        :param samples_name: 存盘的特征文件的代号
        :return:
        '''
        if (not self.model_name in train_filename) or (not self.model_name in test_filename) \
                or (not self.patch_type in train_filename) or (not self.patch_type in test_filename):
            return

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, test_filename)
        D = np.load(data_path)
        test_features = D['arr_0']
        # test_features = test_features[:, np.newaxis]
        test_label = D['arr_1']
        test_label = test_label[:, np.newaxis]
        test_label = to_categorical(test_label, 2)

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, train_filename)
        D = np.load(data_path)
        train_features = D['arr_0']
        # train_features = train_features[:, np.newaxis]
        train_label = D['arr_1']
        train_label = train_label[:, np.newaxis]
        train_label = to_categorical(train_label, 2)

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.h5"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                          epsilon=0.0001, cooldown=0, min_lr=0)
        top_model = self.load_model(mode = 2)
        # optimizer = RMSprop(lr=1e-4, rho=0.9)
        optimizer = SGD(lr=1e-3, momentum=0.9)

        top_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # top_model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss=categorical_crossentropy2, metrics=['accuracy'])

        # print(model.summary())
        # train the model on the new data for a few epochs
        top_model.fit(train_features, train_label, batch_size = batch_size, epochs=epochs,verbose=1,
                      # callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      callbacks=[cp_callback, early_stopping, reduce_lr],
                      validation_data=(test_features, test_label),initial_epoch=initial_epoch)

    def merge_save_model(self):
        checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        latest_top = util.latest_checkpoint(checkpoint_dir)

        full_model = self.create_initial_model()
        full_model.load_weights(latest_top, by_name=True)

        best_model_path = "{}/models/{}_{}_merge_best.h5".format(self._params.PROJECT_ROOT, self.model_name,
                                                           self.patch_type)
        full_model.save(best_model_path, include_optimizer=False)

        # plot_model(full_model, to_file='{}/models/{}.png'.format(self._params.PROJECT_ROOT, self.model_name))

    # def merge_save_model(self):
    #     '''
    #     将新训练的全连接层与 迁移的网络模型进行合并
    #     :param model_dir:新训练的全连接层的checkpoint文件所在目录
    #     :return:
    #     '''
    #
    #     top_model = self.load_model(mode = 2)
    #
    #     model = self.load_model(mode = 0, weights_file="imagenet")
    #
    #     layers_set = ["t_Dense_1", "t_Dense_2"]
    #     for layer_name in layers_set:
    #         new_layer = model.get_layer(name=layer_name)
    #         old_layer = top_model.get_layer(name=layer_name)
    #         weights = old_layer.get_weights()
    #         new_layer.set_weights(weights)
    #
    #     model_dir = "{}/models/trained".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
    #     model_path = model_dir + "/{}_{}.ckpt".format(self.model_name, self.patch_type)
    #     # model.save_weights(model_path, save_format='tf')
    #     model.save_weights(model_path)
    #
    #     plot_model(model, to_file = '{}/models/{}.png'.format(self._params.PROJECT_ROOT, self.model_name))
    #
    #     return model

    def evaluate_entire_model(self, samples_name, batch_size):
        '''
        使用图块文件，评估 合并后的网络
        :param samples_name:图块文件的列表的代号
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation = (False, False))
        model = self.load_model(mode = 0)
        model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        test_loss, test_acc = model.evaluate_generator(test_gen, steps = None, verbose=1, workers=NUM_WORKERS)
        print('Test loss:', test_loss, 'Test accuracy:', test_acc)

        train_loss, train_acc = model.evaluate_generator(train_gen, steps = None, verbose=1, workers=NUM_WORKERS)
        print('Train loss:', train_loss, 'Train accuracy:', train_acc)
        # result = model.predict_generator(test_gen, steps=10)
        # print(result)

    def predict(self, model, src_img, scale, patch_size, seeds):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''

        result = []
        for x, y in seeds:
            block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(ImageNormalization.normalize_mean(img))
            x = np.expand_dims(x, axis=0)
            # x = preprocess_input(x) //训练时没有使用预处理，这里也不能调用

            predictions = model.predict(x)
            class_id = np.argmax(predictions[0])
            probability = predictions[0][class_id]
            result.append((class_id, probability))

        return result

    def predict_on_batch(self, model, src_img, scale, patch_size, seeds, batch_size):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :param batch_size: 每批处理的图片数量
        :return:
        '''
        image_itor = SeedSequence(src_img, scale, patch_size, seeds, batch_size)

        predictions = model.predict_generator(image_itor, verbose=1, workers=NUM_WORKERS)
        result = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            result.append((class_id, probability))

        return result
