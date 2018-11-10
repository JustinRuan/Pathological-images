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

from core.util import read_csv_file
from core import *


class Transfer(object):

    def __init__(self, params):
        self._params = params

        return

    def extract_features(self, src_img, scale, patch_size, seeds):
        '''
        从切片中提取图块，并用网络提取特征
        :param src_img: 切片图像
        :param scale: 提取的倍镜数
        :param patch_size: 图块大小
        :param seeds: 图块中心点的坐标
        :return: 特征
        '''
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # print(base_model.summary())
        features = []
        for x, y in seeds:
            block= src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            feature = base_model.predict(x)
            features.append(feature)

        return features

    def load_model(self, model_path, is_checkpoint = True):
        '''
        从文件中加载网络模型
        :param model_path: 模型的文件名，包括models文件夹中的子文件夹的路径
        :param is_checkpoint: 是否为checkpoint文件
        :return: 网络模型
        '''
        #从checkpoint文件中加载 最后的全连接层的权重
        if is_checkpoint:
            checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, model_path)
            latest = tf.train.latest_checkpoint(checkpoint_dir)

            if latest is None:
                # create the base pre-trained model
                base_model = InceptionV3(weights='imagenet', include_top=False)
            else:
                base_model = InceptionV3(weights=None, include_top=False)

            # add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1")(x)
            # and a logistic layer -- let's say we have 2 classes
            predictions = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_2")(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)
            if not latest is None:
                print("loading >>> ", latest, " ...")
                model.load_weights(latest)
        else:
            # 直接从.h5模型文件中加载整个网络和权重
            model_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, model_path)
            model = load_model(model_dir)

        return model

    def load_data(self, samples_name, batch_size, augmentation = False):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :param batch_size: 图片读取时的每批的图片数量
        :return:用于train和test的两个Sequence
        '''
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size, augmentation)
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size, augmentation)
        return  train_gen, test_gen

    # def fine_tuning_311(self, samples_name):
    #     train_gen, test_gen = self.load_data(samples_name, 20)
    #
    #     # include the epoch in the file name. (uses `str.format`)
    #     checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, "InceptionV3")
    #     # checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
    #     checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"
    #
    #     cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #         checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
    #         # Save weights, every 5-epochs.
    #         period=1)
    #
    #     model = self.load_model("InceptionV3")
    #
    #     # first: train only the top layers (which were randomly initialized)
    #     # i.e. freeze all convolutional InceptionV3 layers
    #     for i, layer in enumerate(model.layers[:311]):
    #         layer.trainable = False
    #         print(i, layer.name, "freezed", sep="\t") # 有311层
    #
    #     # compile the model (should be done *after* setting layers to non-trainable)
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     # print(model.summary())
    #     # train the model on the new data for a few epochs
    #     model.fit_generator(train_gen, steps_per_epoch=3, epochs=1, verbose=1,
    #                         callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
    #                         validation_data=test_gen, validation_steps=3)
    #
    #     # # at this point, the top layers are well trained and we can start fine-tuning
    #     # # convolutional layers from inception V3. We will freeze the bottom N layers
    #     # # and train the remaining top layers.
    #     #
    #     # # let's visualize layer names and layer indices to see how many layers
    #     # # we should freeze:
    #     # for i, layer in enumerate(base_model.layers):
    #     #     print(i, layer.name)
    #     #
    #     # # we chose to train the top 2 inception blocks, i.e. we will freeze
    #     # # the first 249 layers and unfreeze the rest:
    #     # for layer in model.layers[:249]:
    #     #     layer.trainable = False
    #     # for layer in model.layers[249:]:
    #     #     layer.trainable = True
    #     #
    #     # # we need to recompile the model for these modifications to take effect
    #     # # we use SGD with a low learning rate
    #     # from tensorflow.keras.optimizers import SGD
    #     # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    #     #
    #     # # we train our model again (this time fine-tuning the top 2 inception blocks
    #     # # alongside the top Dense layers
    #     # model.fit_generator(train_gen, steps_per_epoch=100, epochs=1, verbose=1, callbacks = [cp_callback],
    #     #                     validation_data=test_gen, validation_steps=100)

    def fine_tuning_model(self, model_dir, samples_name, freezed_num, optimizer):
        '''
        微调模型，训练时冻结网络中0到freezed_num层
        :param model_dir:训练时checkpoint文件的存盘路径
        :param samples_name:训练所使用的图片列表文件的代号
        :param freezed_num: 训练时冻结网络中 0 到 freezed_num 层
        :param optimizer: 训练用，自适应学习率算法
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, 20)

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, model_dir)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        model = self.load_model(model_dir)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for i, layer in enumerate(model.layers[:freezed_num]):
            layer.trainable = False
            print( " freezed ", i, layer.name, sep="\t\t")
        for i, layer in enumerate(model.layers[freezed_num:]):
            layer.trainable = True
            print("trainable", i + freezed_num, layer.name, sep="\t\t")

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # print(model.summary())
        # train the model on the new data for a few epochs
        model.fit_generator(train_gen, steps_per_epoch=3, epochs=1, verbose=1,
                            callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            validation_data=test_gen, validation_steps=3)
        return

    def fine_tuning_1(self, samples_name):
        '''
        只训练全连接层
        :param samples_name:
        :return:
        '''
        self.fine_tuning_model("InceptionV3", samples_name, 311, RMSprop(lr=1e-4, rho=0.9))

    def fine_tuning_2(self, samples_name):
        '''
        训练全连接层，和最后一部分的原网络
        :param samples_name:
        :return:
        '''
        self.fine_tuning_model("InceptionV3_2", samples_name, 249, SGD(lr=0.0001, momentum=0.9))

    def predict(self, src_img, scale, patch_size, seeds):
        '''
        预测在种子点提取的图块
        :param src_img: 切片图像
        :param scale: 提取图块的倍镜数
        :param patch_size: 图块大小
        :param seeds: 种子点的集合
        :return:
        '''
        model = self.load_model("InceptionV3/V3-0.10-0.96.h5", False)
        # model = self.merge_model("InceptionV3_2")
        print(model.summary())

        result = []
        for x, y in seeds:
            block = src_img.get_image_block(scale, x, y, patch_size, patch_size)
            img = block.get_img()

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            # x = preprocess_input(x) //训练时没有使用预处理，这里也不能调用

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
        :param batch_size: 每批处理的图片数量
        :return:
        '''
        model = self.load_model("InceptionV3/V3-0.06-0.99.h5", False)
        # model = self.merge_model("InceptionV3_2")
        print(model.summary())

        image_itor = SeedSequence(src_img, scale, patch_size, seeds, batch_size)

        predictions = model.predict_generator(image_itor, verbose=1)
        result = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            result.append((class_id, probability))

        return result

    def extract_features_for_train(self, samples_name):
        '''
        从训练的图块中提取 特征， 并存盘
        :param samples_name: 图块文件所在列表txt文件
        :return: 生成两个特征文件
        '''
        batch_size = 20
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        data_path = "{}/data/{}_".format(self._params.PROJECT_ROOT, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size)
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size)

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        features_layer = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=features_layer)

        step_count = len(Ytest) // batch_size
        # step_count = 10
        test_features = model.predict_generator(test_gen, steps=step_count, verbose=1)
        test_label = Ytest[:step_count * batch_size]
        np.savez(data_path + "features_test", test_features, test_label)

        step_count = len(Ytrain) // batch_size
        # step_count = 10
        train_features = model.predict_generator(train_gen, steps=step_count, verbose=1)
        train_label = Ytrain[:step_count * batch_size]
        np.savez(data_path + "features_train", train_features, train_label)
        return

    def fine_tuning_saved_file(self,samples_name):
        '''
        使用存盘的特征文件来训练 全连接层
        :param samples_name: 存盘的特征文件的代号
        :return:
        '''
        data_path = "{}/data/{}_".format(self._params.PROJECT_ROOT, samples_name)
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

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, "InceptionV3_2")
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.2f}-{val_acc:.2f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        top_model = Sequential()
        top_model.add(Flatten(input_shape=(1, 2048)))
        top_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1"))
        top_model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_2"))

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest is None:
            print("loading >>> ", latest, " ...")
            top_model.load_weights(latest)

        top_model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        # print(model.summary())
        # train the model on the new data for a few epochs
        top_model.fit(train_features, train_label, batch_size =200, epochs=20,
                      callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      validation_data=(test_features, test_label))

    def merge_model(self, model_dir):
        '''
        将新训练的全连接层与 迁移的网络模型进行合并
        :param model_dir:新训练的全连接层的checkpoint文件所在目录
        :return:
        '''
        checkpoint_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, model_dir)
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if latest is None: return None

        top_model = Sequential()
        top_model.add(Flatten(input_shape=(1, 2048)))
        top_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1"))
        top_model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_2"))
        print("loading >>> ", latest, " ...")
        top_model.load_weights(latest)

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_1")(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name="t_Dense_2")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        layers_set = ["t_Dense_1", "t_Dense_2"]
        for layer_name in layers_set:
            new_layer = model.get_layer(name=layer_name)
            old_layer = top_model.get_layer(name=layer_name)
            weights = old_layer.get_weights()
            new_layer.set_weights(weights)

        return model

    def evaluate_merged_model(self, samples_name):
        '''
        使用图块文件，评估 合并后的网络
        :param samples_name:图块文件的列表的代号
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, 20, augmentation = False)

        model = self.merge_model("InceptionV3_2")
        model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        test_loss, test_acc = model.evaluate_generator(test_gen, steps = 10)

        model_dir = "{}/models/{}".format(self._params.PROJECT_ROOT, "InceptionV3")
        model_path = model_dir + "/V3-{:.2f}-{:.2f}.h5".format(test_loss, test_acc)
        model.save(model_path)

        print('Test accuracy:', test_acc)

        # result = model.predict_generator(test_gen, steps=10)
        # print(result)
