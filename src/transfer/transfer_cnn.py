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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras import backend as K
from keras.losses import categorical_crossentropy

from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout, Activation
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adamax, Nadam, Adadelta
from keras.preprocessing import image
from keras.utils import to_categorical, plot_model
from skimage.transform import resize

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn import metrics

from preparation.normalization import ImageNormalization
from core.util import read_csv_file
from core import *
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from statistics import mode

NUM_CLASSES = 2

# def categorical_crossentropy2(y_true, y_pred):
#   return K.categorical_crossentropy(y_true, y_pred)

class Transfer(object):

    def __init__(self, params, model_name, patch_type):
        '''
        初始化各种参数
        :param params: 外部参数
        :param model_name: 当前所使用的迁移模型的名称
        :param patch_type: 所使用的图块的倍镜和大小：
                         # patch_type 500_128, 2000_256, 4000_256
        '''
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        if self.model_name == "inception_v3":
            self.input_image_size = 299 #299
        else:
            self.input_image_size = 224

        return

    def extract_features(self,base_model, src_img, scale, patch_size, seeds):
        '''
        从切片中提取图块，并用网络提取特征
        :param base_model: 加载的模型
        :param src_img: 切片图像
        :param scale: 提取的倍镜数
        :param patch_size: 图块大小
        :param seeds: 图块中心点的坐标
        :return: 特征数据
        '''
        # create the base pre-trained model
        # base_model = InceptionV3(weights='imagenet', include_top=False, pooling = 'avg')
        # base_model = DenseNet121(weights='imagenet', include_top=False, pooling = 'avg')
        # base_model = ResNet50(weights='imagenet', include_top=False, pooling = 'avg')
        print(base_model.summary())
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

    def add_new_top_layers(self, input_layer):
        '''
        生成Top的全连接层
        :param input_layer: 输入
        :return:
        '''
        # let's add a fully-connected layer, kernel_regularizer=regularizers.l2(0.01),
        x = Dense(1024, activation='relu', name="top_Dense")(input_layer)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)
        return predictions

    def create_initial_model(self):
        '''
        使用由imagenet所训练的权重来初始化网络，不包括原来的top层
        并加入全局平均池化和新的全连接层
        :return: 完整的模型
        '''
        if self.model_name == "inception_v3":
            base_model = InceptionV3(weights='imagenet', include_top=False)
        elif self.model_name == "densenet121":
            base_model = DenseNet121(weights='imagenet', include_top=False)
        elif self.model_name == "densenet169":
            base_model = DenseNet169(weights='imagenet', include_top=False)
        elif self.model_name == "densenet201":
            base_model = DenseNet201(weights='imagenet', include_top=False)
        elif self.model_name == "resnet50":
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif self.model_name == "inception_resnet_v2":
            base_model = InceptionResNetV2(weights='imagenet', include_top=False)
        elif self.model_name == "vgg16":
            base_model = VGG16(weights='imagenet', include_top=False)
        elif self.model_name == "mobilenet_v2":
            base_model = MobileNetV2(weights='imagenet', include_top=False)
        # elif self.model_name == "nasnet": # 这个有问题： Exception: Incompatible shapes: [100,22,15,15] vs. [100,22,31,31]
        #     base_model = NASNetMobile(weights=None, include_top=False)
        #     weights_path = os.path.join(os.path.expanduser('~'), '.keras/models/nasnet_mobile_no_top.h5')
        #     base_model.load_weights(weights_path, by_name=True)

        top_avg_pool = GlobalAveragePooling2D(name='top_avg_pool')(base_model.output)
        predictions = self.add_new_top_layers(top_avg_pool)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def load_model(self, mode, model_file = None):
        '''
        加载模型
        :param mode: 加载的方式
        :param model_file: 已经有的模型文件
        :return:
        '''
        if mode == 999: # 直接加载模型文件
            print("loading >>> ", model_file, " ...")
            model = load_model(model_file)
            return model

        elif mode == 0:  # "load_entire_model, 优先加载checkpoint_dir下最新的模型文件，用于整个网络微调时"
            checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = util.latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                model = load_model(latest)
            else:
                if model_file is not None and os.path.exists(model_file):
                    print("loading >>> ", model_file, " ...")
                    model = load_model(model_file)
                else:
                    model = self.create_initial_model()

            return model

        elif mode == 1:  # "extract features for transfer learning"

            full_model = self.create_initial_model()
            feature_model = Model(inputs=full_model.input, outputs=full_model.get_layer('top_avg_pool').output)
            return feature_model

        elif mode == 2: # 仅训练top层
            checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest_top = util.latest_checkpoint(checkpoint_dir)
            if latest_top is not None:
                top_model = load_model(latest_top, compile=False)
            else:
                features_num = {"inception_v3": 2048,
                                "densenet121": 1024,
                                "densenet169": 1664,
                                "densenet201": 1920,
                                "resnet50": 2048,
                                "inception_resnet_v2": 1536,
                                "vgg16": 512,
                                "mobilenet_v2": 1280}

                input_layer = Input(shape=(features_num[self.model_name],))
                predictions = self.add_new_top_layers(input_layer)
                top_model = Model(inputs=input_layer, outputs=predictions)
            return top_model

    def load_data(self, samples_name, batch_size, augmentation = (False, False)):
        '''
        从图片的列表文件中加载数据，到Sequence中
        :param samples_name: 列表文件的代号
        :param batch_size: 图片读取时的每批的图片数量
        :param augmentation: 训练集和测试集是否进行实时的扩增
        :return:
        '''
        train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, train_list)
        train_gen = ImageSequence(Xtrain, Ytrain, batch_size, self.input_image_size, NUM_CLASSES, augmentation[0])
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        test_gen = ImageSequence(Xtest, Ytest, batch_size, self.input_image_size, NUM_CLASSES, augmentation[1])
        return  train_gen, test_gen


    ##############################################################################################################
    #        全网络的微调过程
    ##############################################################################################################
    def fine_tuning_model(self, optimizer, samples_name, batch_size, freezed_num, epochs=20, initial_epoch=0):
        '''
        微调全网模型，训练时冻结网络中0到freezed_num层
        :param optimizer: 优化器
        :param samples_name: 训练所使用的图片列表文件的代号
        :param batch_size: 图片读取时的每批的图片数量
        :param freezed_num:  训练时冻结网络中 0 到 freezed_num 层
        :param epochs: 当前训练所设定的epochs
        :param initial_epoch: 本次训练的起点epochs
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size)

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5"

        merged_model = "{}/models/{}_{}_merge_best.h5".format(self._params.PROJECT_ROOT, self.model_name,
                                                           self.patch_type)
        model = self.load_model(mode = 0, model_file= merged_model)

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
        model.fit_generator(train_gen, steps_per_epoch=train_step, epochs=epochs, verbose=1, workers=self.NUM_WORKERS,
                            # callbacks = [cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                            callbacks=[cp_callback, early_stopping, reduce_lr],
                            validation_data=test_gen, validation_steps=test_step, initial_epoch = initial_epoch)
        return


    def fine_tuning_model_with_freezed(self, samples_name, batch_size, freezed_num, epochs, initial_epoch):
        '''
        训练整个网络，可以冻结前面部分，只训练全连接层，和最后一部分的原网络
        :param samples_name: 训练所使用的图片列表文件的代号
        :param batch_size: 图片读取时的每批的图片数量
        :param freezed_num: 训练时冻结网络中 0 到 freezed_num 层
        :param epochs: 当前训练所设定的epochs
        :param initial_epoch: 本次训练的起点epochs
        :return:
        '''
        # optimizer = SGD(lr=1e-4, momentum=0.9)
        # optimizer = RMSprop(lr=1e-4, rho=0.9)
        optimizer = SGD(lr=1e-3, momentum=0.9)
        # 最后两个Inception的位置：249, 最后一个的位置：280, Top的位置：311
        self.fine_tuning_model(optimizer, samples_name, batch_size, freezed_num, epochs, initial_epoch)

    #################################################################################################################
    #     提取图块的特征并存盘，只训练Top部分的过程
    #################################################################################################################

    def extract_features_for_train(self, samples_name, batch_size):
        '''
        从训练的图块中提取 特征， 并存盘
        :param samples_name: 图块文件所在列表txt文件
        :param batch_size: 每个批次中图片的数量
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
        '''
        读取文件列表，加载图块提取 特征， 并存盘
        :param model: 提取特征所使用的网络模型
        :param samples_file_path: 图块列表文件的文件名
        :param batch_size: 每个批次的图片数量
        :param augmentation: 是否进行实时的数据提升
        :param aug_multiple: 当进行提升时，提升后的数据集与原来数据集的倍数关系
        :return: 将提取特征进行存盘
        '''
        sample_name = samples_file_path[:-4]
        file_list = "{}/{}".format(self._params.PATCHS_ROOT_PATH, samples_file_path)
        if not augmentation:
            save_path = "{}/data/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, sample_name)
        else:
            save_path = "{}/data/{}_{}_Aug{}".format(self._params.PROJECT_ROOT, self.model_name,
                                                            sample_name, aug_multiple)

        X, Y = read_csv_file(self._params.PATCHS_ROOT_PATH, file_list)
        data_gen = ImageSequence(X, Y, batch_size, self.input_image_size, augmentation)

        if not augmentation:
            aug_multiple = 1

        step_count = int(aug_multiple * len(Y) / batch_size)

        features = model.predict_generator(data_gen, steps=step_count, verbose=1,workers=self.NUM_WORKERS)
        labels = Y[:len(features)]
        np.savez(save_path + "_features", features, labels)
        return

    def fine_tuning_top_cnn_model_saved_file(self, train_filename, test_filename,
                                         batch_size = 100, epochs = 20, initial_epoch = 0):
        '''
         使用存盘的特征文件来训练 全连接层
        :param train_filename: 训练集的文件
        :param test_filename: 测试集的文件
        :param batch_size: 每个批次使用的数据量
        :param epochs: 当前训练所设定的epochs
        :param initial_epoch: 本次训练的起点epochs
        :return:
        '''
        if (not self.model_name in train_filename) or (not self.model_name in test_filename) \
                or (not self.patch_type in train_filename) or (not self.patch_type in test_filename):
            return

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, test_filename)
        D = np.load(data_path)
        test_features = D['arr_0']
        test_label = D['arr_1']
        test_label = test_label[:, np.newaxis]
        test_label = to_categorical(test_label, NUM_CLASSES)

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, train_filename)
        D = np.load(data_path)
        train_features = D['arr_0']
        train_label = D['arr_1']
        train_label = train_label[:, np.newaxis]
        train_label = to_categorical(train_label, NUM_CLASSES)

        # include the epoch in the file name. (uses `str.format`)
        checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False,
            period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                          epsilon=0.0001, cooldown=0, min_lr=0)
        top_model = self.load_model(mode = 2)

        optimizer = SGD(lr=1e-3, momentum=0.9)

        top_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # print(model.summary())
        # train the model on the new data for a few epochs
        top_model.fit(train_features, train_label, batch_size = batch_size, epochs=epochs,verbose=1,
                      # callbacks=[cp_callback, TensorBoard(log_dir=checkpoint_dir)],
                      callbacks=[cp_callback, early_stopping, reduce_lr],
                      validation_data=(test_features, test_label),initial_epoch=initial_epoch)

    def merge_save_model(self):
        '''
        当训练好的Top部分与前端的网络进行融合，并存盘
        :return:
        '''
        checkpoint_dir = "{}/models/{}_{}_top".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)
        latest_top = util.latest_checkpoint(checkpoint_dir)

        full_model = self.create_initial_model()
        full_model.load_weights(latest_top, by_name=True)

        best_model_path = "{}/models/{}_{}_merge_cnn.h5".format(self._params.PROJECT_ROOT, self.model_name,
                                                           self.patch_type)
        full_model.save(best_model_path, include_optimizer=False)

        # plot_model(full_model, to_file='{}/models/{}.png'.format(self._params.PROJECT_ROOT, self.model_name))

    def evaluate_entire_cnn_model(self, samples_name, batch_size, model_file = None):
        '''
        使用图块文件，评估 合并后的网络
        :param samples_name: 图块文件的列表的代号
        :param batch_size:每个批次所使用的图片数量
        :param model_file: 指定的模型文件
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation = (False, False))
        model = self.load_model(mode = 999, model_file=model_file)
        model.compile(optimizer=RMSprop(lr=1e-4, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        test_loss, test_acc = model.evaluate_generator(test_gen, steps = None, verbose=1, workers=self.NUM_WORKERS)
        print('Test loss:', test_loss, 'Test accuracy:', test_acc)

        train_loss, train_acc = model.evaluate_generator(train_gen, steps = None, verbose=1, workers=self.NUM_WORKERS)
        print('Train loss:', train_loss, 'Train accuracy:', train_acc)
        # result = model.predict_generator(test_gen, steps=10)
        # print(result)

    def evaluate_cnn_svm_rf_model(self, samples_name, batch_size):
        '''
        使用CNN提取特征，分别用NN, SVM, RF分类器进行分类测试，并完成集成的投票功能
        :param samples_name: 特征文件的代号
        :param batch_size: 每批次的图片数量
        :return:
        '''
        train_gen, test_gen = self.load_data(samples_name, batch_size, augmentation = (False, False))
        feature_model = self.load_model(mode = 1)
        feature_model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])

        output_features = feature_model.predict_generator(test_gen, steps=None, verbose=1)

        cnn_model = self.load_model(mode = 2)
        cnn_model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])
        result = cnn_model.predict(output_features)

        y_cnn_pred = []
        for pred_dict in result:
            class_id = np.argmax(pred_dict)
            y_cnn_pred.append(class_id)

        svm_model_file = self._params.PROJECT_ROOT + "/models/svm_{}_{}.model".format(self.model_name, self.patch_type)
        clf = joblib.load(svm_model_file)
        y_svm_pred = clf.predict(output_features)

        rf_model_file = self._params.PROJECT_ROOT + "/models/rf_{}_{}.model".format(self.model_name, self.patch_type)
        clf = joblib.load(rf_model_file)
        y_rf_pred = clf.predict(output_features)

        test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        Xtest, Ytest = read_csv_file(self._params.PATCHS_ROOT_PATH, test_list)
        y_true = Ytest[:len(y_cnn_pred)]
        cnn_score = metrics.accuracy_score(y_true, y_cnn_pred)
        svm_score = metrics.accuracy_score(y_true, y_svm_pred)
        rf_score = metrics.accuracy_score(y_true, y_rf_pred)
        print("cnn score = ", cnn_score)
        print( "svm score = ", svm_score)
        print("rf score = ", rf_score)

        voting_result = []
        for y_cnn, y_svm, y_rf in zip(y_cnn_pred, y_svm_pred, y_rf_pred):
            most = mode([y_cnn, y_svm, y_rf])
            voting_result.append(most)

        voting_score = metrics.accuracy_score(y_true, voting_result)
        print("voting score = ", voting_score)

    def ensemble_predcit(self):

        return

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

            x = image.img_to_array(resize(ImageNormalization.normalize_mean(img)),
                                   (self.input_image_size, self.input_image_size))
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
        image_itor = SeedSequence(src_img, scale, patch_size, self.input_image_size, seeds, batch_size)

        predictions = model.predict_generator(image_itor, verbose=1, workers=self.NUM_WORKERS)
        result = []
        for pred_dict in predictions:
            class_id = np.argmax(pred_dict)
            probability = pred_dict[class_id]
            result.append((class_id, probability))

        return result



    #################################################################################################
    #              SVM
    ##################################################################################################

    def train_top_svm(self, train_filename, test_filename):
        '''
        训练SVM分类器，作为CNN网络的新TOP。 包含参数寻优过程
        :param train_filename: 保存特征的训练集文件
        :param test_filename: 测试集文件
        :return:
        '''
        if (not self.model_name in train_filename) or (not self.model_name in test_filename) \
                or (not self.patch_type in train_filename) or (not self.patch_type in test_filename):
            return

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, test_filename)
        D = np.load(data_path)
        test_features = D['arr_0']
        test_label = D['arr_1']

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, train_filename)
        D = np.load(data_path)
        train_features = D['arr_0']
        train_label = D['arr_1']

        max_iter = 500
        model_params = [ {'C':0.0001}, {'C':0.001 }, {'C':0.01}, {'C':0.1},
                         {'C':0.5}, {'C':1.0}, {'C':1.2}, {'C':1.5},
                         {'C':2.0}, {'C':10.0} ]
        # model_params = [{'C': 0.0001}]
        # K_num = [100, 200, 300, 500, 1024, 2048]
        #
        result = {'pred': None, 'score': 0, 'clf': None}
        # for item in K_num:
        #     sb = SelectKBest(k=item).fit(train_features, train_label)
        #     train_x_new = sb.transform(train_features)
        #     test_x_new = sb.transform(test_features)

        # 进行了简单的特征选择，选择全部特征。
        # inception_v3 ： the best score = 0.8891836734693878, k = 2048， C=0.0001
        # densenet121: the best score = 0.9151020408163265, k=1024, C=0.01
        # resnet50: the best score = 0.8187755102040817, C=0.5
        feature_num = len(train_features[0])
        for params in model_params:
            clf = LinearSVC(**params, max_iter=max_iter, verbose=0)
            clf.fit(train_features, train_label)
            y_pred = clf.predict(test_features)
            score = metrics.accuracy_score(test_label, y_pred)
            print('feature num = {}, C={:8f} => score={:5f}'.format(feature_num, params['C'], score))

            if score > result["score"]:
                result = {'pred': y_pred, 'score': score, 'clf': clf}

        print("the best score = {}".format(result["score"]))

        print("Classification report for classifier %s:\n%s\n"
              % (result["clf"], metrics.classification_report(test_label, result["pred"])))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label, result["pred"]))

        model_file = self._params.PROJECT_ROOT + "/models/svm_{}_{}.model".format(self.model_name, self.patch_type)
        joblib.dump(result["clf"], model_file)
        return

    #################################################################################################
    #              Random Forest
    ##################################################################################################
    def train_top_rf(self, train_filename, test_filename):
        '''
        训练RF分类器，作为CNN网络的新TOP。 包含参数寻优过程
        :param train_filename: 保存特征的训练集文件
        :param test_filename: 测试集文件
        :return:
        '''
        if (not self.model_name in train_filename) or (not self.model_name in test_filename) \
                or (not self.patch_type in train_filename) or (not self.patch_type in test_filename):
            return

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, test_filename)
        D = np.load(data_path)
        test_features = D['arr_0']
        test_label = D['arr_1']

        data_path = "{}/data/{}".format(self._params.PROJECT_ROOT, train_filename)
        D = np.load(data_path)
        train_features = D['arr_0']
        train_label = D['arr_1']

        # # ###########################  参数寻优  #############################
        param_grid = [
            {'n_estimators': [50, 100, 150, 200], "max_depth":[10, 20]},
            # {'n_estimators': [200], 'criterion': ['gini'],'min_impurity_decrease': np.linspace(0,0.5, 20)}
            # {'n_estimators': [200], 'min_samples_split': range(10, 100, 10)}
        ]

        clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=2, n_jobs = self.NUM_WORKERS)
        clf.fit(train_features, train_label)

        print("the best score = {}".format(clf.best_score_))
        print("the best param = {}".format(clf.best_params_))

        # 5 x 128
        # # inception_v3 the best score = 0.8675879396984925， param = {'max_depth': 20, 'n_estimators': 150}
        # # densenet121: the best score = 0.911608040201005, param = {'max_depth': 20, 'n_estimators': 150}
        # # resnet50: the best score = 0.8113065326633165, param = {'max_depth': 20, 'n_estimators': 200}
        # clf = RandomForestClassifier(n_estimators = 150, max_depth=20, n_jobs=self.NUM_WORKERS)
        # clf = clf.fit(train_features, train_label)
        # y_pred = clf.predict(test_features)
        # score = metrics.accuracy_score(test_label, y_pred)
        # print("score = ", score)

        model_file = self._params.PROJECT_ROOT + "/models/rf_{}_{}.model".format(self.model_name, self.patch_type)
        joblib.dump(clf, model_file)