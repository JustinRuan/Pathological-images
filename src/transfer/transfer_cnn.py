#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-30'

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from preparation.normalization import ImageNormalization
from core.util import read_csv_file
from core import *

NUM_CLASSES = 2
NUM_WORKERS = 1

class Transfer(object):
    # patch_type 500_128, 2000_256
    def __init__(self, params, model_name, patch_type):
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        return

    # def extract_features(self, src_img, scale, patch_size, seeds):
    #     '''
    #     从切片中提取图块，并用网络提取特征
    #     :param src_img: 切片图像
    #     :param scale: 提取的倍镜数
    #     :param patch_size: 图块大小
    #     :param seeds: 图块中心点的坐标
    #     :return: 特征
    #     '''
    #     # create the base pre-trained model
    #     base_model = InceptionV3(weights='imagenet', include_top=False)
    #     # print(base_model.summary())
    #     features = []
    #     for x, y in seeds:
    #         block= src_img.get_image_block(scale, x, y, patch_size, patch_size)
    #         img = block.get_img()
    #
    #         x = image.img_to_array(img)
    #         x = np.expand_dims(x, axis=0)
    #         x = preprocess_input(x)
    #
    #         feature = base_model.predict(x)
    #         features.append(feature)
    #
    #     return features

    # patch_type 500_128, 2000_256
    def load_model(self, mode, weights_file = None):

        if self.model_name == "inception_v3":
            if mode == 0: # "load_entire_model":
                if weights_file is None:
                    base_model = InceptionV3(weights='imagenet', include_top=False)
                else:
                    base_model = InceptionV3(weights=None, include_top=False)

                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                # let's add a fully-connected layer
                x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1")(x)
                # and a logistic layer -- let's say we have 2 classes
                predictions = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                                    name="t_Dense_2")(x)

                # this is the model we will train
                model = Model(inputs=base_model.input, outputs=predictions)

                if not weights_file is None:
                    model_path = "{}/models/{}".format(self._params.PROJECT_ROOT, weights_file)
                    print("loading >>> ", model_path, " ...")
                    model.load_weights(model_path)
                    return model
                else:
                    return model

            elif mode == 1: # "extract features for transfer learning"
                base_model = InceptionV3(weights='imagenet', include_top=False)
                x = base_model.output
                features_layer = GlobalAveragePooling2D()(x)
                model = Model(inputs=base_model.input, outputs=features_layer)
                return model

            elif mode == 2: # "refine top model"
                top_model = Sequential()

                top_model.add(Flatten(input_shape=(1, 2048)))
                top_model.add(
                    Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1"))
                top_model.add(
                    Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_2"))

                checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
                latest = tf.train.latest_checkpoint(checkpoint_dir)
                if not latest is None:
                    print("loading >>> ", latest, " ...")
                    top_model.load_weights(latest)

                return top_model

        # elif model_name == "others xxxxx":
        #
        #     return

        return None

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


    # def fine_tuning_model(self, model_dir, samples_name, freezed_num, optimizer):
    #     '''
    #     微调模型，训练时冻结网络中0到freezed_num层
    #     :param model_dir:训练时checkpoint文件的存盘路径
    #     :param samples_name:训练所使用的图片列表文件的代号
    #     :param freezed_num: 训练时冻结网络中 0 到 freezed_num 层
    #     :param optimizer: 训练用，自适应学习率算法
    #     :return:
    #     '''
    #     train_gen, test_gen = self.load_data(samples_name, 20)
    #
    #     # include the epoch in the file name. (uses `str.format`)
    #     checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, model_dir)
    #     checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"
    #
    #     cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #         checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
    #         # Save weights, every 5-epochs.
    #         period=1)
    #
    #     model = self.load_model(model_dir)
    #
    #     # first: train only the top layers (which were randomly initialized)
    #     # i.e. freeze all convolutional InceptionV3 layers
    #     for i, layer in enumerate(model.layers[:freezed_num]):
    #         layer.trainable = False
    #         print( " freezed ", i, layer.name, sep="\t\t")
    #     for i, layer in enumerate(model.layers[freezed_num:]):
    #         layer.trainable = True
    #         print("trainable", i + freezed_num, layer.name, sep="\t\t")
    #
    #     # compile the model (should be done *after* setting layers to non-trainable)
    #     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     # print(model.summary())
    #     # train the model on the new data for a few epochs
    #     model.fit_generator(train_gen, steps_per_epoch=3, epochs=1, verbose=1,
    #                         callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
    #                         validation_data=test_gen, validation_steps=3)
    #     return
    #
    # def fine_tuning_1(self, samples_name):
    #     '''
    #     只训练全连接层
    #     :param samples_name:
    #     :return:
    #     '''
    #     self.fine_tuning_model("InceptionV3", samples_name, 311, RMSprop(lr=1e-4, rho=0.9))
    #
    # def fine_tuning_2(self, samples_name):
    #     '''
    #     训练全连接层，和最后一部分的原网络
    #     :param samples_name:
    #     :return:
    #     '''
    #     self.fine_tuning_model("InceptionV3_2", samples_name, 249, SGD(lr=0.0001, momentum=0.9))

    def extract_features_for_train(self, samples_name, batch_size, augmentation = (False, False),
                                   aug_multiple = (1, 1)):
        '''
        从训练的图块中提取 特征， 并存盘
        :param samples_name: 图块文件所在列表txt文件
        :return: 生成两个特征文件
        '''
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        save_path = "{}/data/{}_{}_".format(self._params.PROJECT_ROOT, self.model_name, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size)
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size)

        model = self.load_model(mode = 1)

        step_count = len(Ytest) // batch_size
        # step_count = 10
        test_features = model.predict_generator(test_gen, steps=step_count, verbose=1,workers=NUM_WORKERS)
        test_label = Ytest[:step_count * batch_size]
        np.savez(save_path + "features_test", test_features, test_label)

        step_count = len(Ytrain) // batch_size
        # step_count = 10
        train_features = model.predict_generator(train_gen, steps=step_count, verbose=1, workers=NUM_WORKERS)
        train_label = Ytrain[:step_count * batch_size]
        np.savez(save_path + "features_train", train_features, train_label)
        return

    def fine_tuning_top_model_saved_file(self, samples_name, batch_size = 100, epochs = 20, initial_epoch = 0):
        '''
        使用存盘的特征文件来训练 全连接层
        :param samples_name: 存盘的特征文件的代号
        :return:
        '''
        data_path = "{}/data/{}_{}_".format(self._params.PROJECT_ROOT, self.model_name, samples_name)
        D = np.load(data_path + "features_test.npz")
        test_features = D['arr_0']
        test_features = test_features[:, np.newaxis]
        test_label = D['arr_1']
        test_label = test_label[:, np.newaxis]
        test_label = to_categorical(test_label, 2)

        D = np.load(data_path + "features_train.npz")
        train_features = D['arr_0']
        train_features = train_features[:, np.newaxis]
        train_label = D['arr_1']
        train_label = train_label[:, np.newaxis]
        train_label = to_categorical(train_label, 2)

        if self.patch_type in samples_name:
            patch_type = self.patch_type
        else:
            patch_type = "error"

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            period=1)

        top_model = self.load_model(mode = 2)

        top_model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        # print(model.summary())
        # train the model on the new data for a few epochs
        top_model.fit(train_features, train_label, batch_size = batch_size, epochs=epochs,verbose=0,
                      callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      validation_data=(test_features, test_label),initial_epoch=initial_epoch)

    def merge_save_model(self):
        '''
        将新训练的全连接层与 迁移的网络模型进行合并
        :param model_dir:新训练的全连接层的checkpoint文件所在目录
        :return:
        '''

        top_model = self.load_model(mode = 2)

        model = self.load_model(mode = 0)

        layers_set = ["t_Dense_1", "t_Dense_2"]
        for layer_name in layers_set:
            new_layer = model.get_layer(name=layer_name)
            old_layer = top_model.get_layer(name=layer_name)
            weights = old_layer.get_weights()
            new_layer.set_weights(weights)

        model_dir = "{}/models/trained".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        model_path = model_dir + "/{}_{}.ckpt".format(self.model_name, self.patch_type)
        model.save_weights(model_path, save_format='tf')

        return model

    def evaluate_entire_model(self, weights_file, samples_name, batch_size):
        '''
        使用图块文件，评估 合并后的网络
        :param samples_name:图块文件的列表的代号
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation = (False, False))
        test_len = test_gen.__len__()

        model = self.load_model(mode = 0, weights_file=weights_file)
        model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        test_loss, test_acc = model.evaluate_generator(test_gen, steps = 10, verbose=1, workers=NUM_WORKERS)

        print('Test accuracy:', test_acc)

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
        # model = self.load_model("InceptionV3/V3-0.10-0.96.h5", False)
        # # model = self.merge_model("InceptionV3_2")
        # print(model.summary())

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
