#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-25'

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from sklearn import metrics
from skimage import io, util
from core import *
from core.util import read_csv_file
from cnn.net import alexnet, googlenet, simpleNet128

# tf.logging.set_verbosity(tf.logging.INFO)

class cnn_tensor(object):

    def __init__(self, params, model_name, samples_name = None):
        '''
        初始化CNN分类器
        :param params: 参数
        :param model_name: 使用的模型文件
        :param samples_name: 使用的标本集的关键字（标记符），为None时是进入预测模式
        '''
        model_name = model_name + "_tensorflow"
        self._params = params
        self.model_name = model_name

        self.model_root = "{}/models/{}/".format(self._params.PROJECT_ROOT, model_name)

        if (not samples_name is None):
            self.train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
            self.test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        return

    def _parse_function(self, filename, label):
        '''
        解析数据，在train和test时，将图像文件读入Tensor环境，并加以标注
        :param filename: 图像文件路径
        :param label: 标注
        :return: 图像Tensor和它的标注
        '''
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_float = tf.cast(image_decoded, dtype=tf.float32)
        return {"x":image_float}, label

    def eval_input_fn(self, csv_path, batch_size):
        '''
        在test时，数据输入
        :param csv_path: test文件的路径
        :param batch_size: 每批处理的数量
        :return:
        '''
        filenames_list, labels_list = read_csv_file(self._params.PATCHS_ROOT_PATH, csv_path)

        # A vector of filenames.
        filenames = tf.constant(filenames_list)

        # `labels[i]` is the label for the image in `filenames[i].
        labels = tf.constant(labels_list)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        return dataset

    def train_input_fn(self, csv_path, batch_size, repeat_count):
        '''
        在train时，数据输入
        :param csv_path: train文件的路径
        :param batch_size: 每批处理的数量
        :param repeat_count: 数据重复使用的次数
        :return:
        '''
        filenames_list, labels_list = read_csv_file(self._params.PATCHS_ROOT_PATH, csv_path)

        # A vector of filenames.
        filenames = tf.constant(filenames_list)

        # `labels[i]` is the label for the image in `filenames[i].
        labels = tf.constant(labels_list)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.repeat(repeat_count).prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        return dataset

    def training(self):
        '''
        训练网络
        :return:
        '''
        GPU = False
        if GPU:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu %的显存
            gpu_config.gpu_options.allow_growth = True  # 程序按需申请内存

            config = tf.estimator.RunConfig(keep_checkpoint_max=3, session_config=gpu_config)
        else:
            config = tf.estimator.RunConfig(keep_checkpoint_max=3)

        if "googlenet" in self.model_name.lower():
            model_fn = googlenet.googlenet_model_fn
        elif "alexnet" in self.model_name.lower():
            model_fn = alexnet.alexnet_model_fn
        elif "simplenet128" in self.model_name.lower():
            model_fn = simpleNet128.simpleNet128_model_fn
        else:
            model_fn = None

        # Create the Estimator
        classifier = tf.estimator.Estimator(
            model_fn= model_fn, model_dir=self.model_root, config=config)

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        # tensors_to_log = {"probabilities": "softmax_tensor", "loss":"loss"}
        tensors_to_log = {"loss": "loss"}

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)

        # Train the model
        batch_size = 100
        classifier.train(
            input_fn=lambda:self.train_input_fn(self.train_list, batch_size, 3),
            steps=1000,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=lambda:self.eval_input_fn(self.test_list, batch_size))
        print(eval_results)

        return

    def prepare_predict(self, src_img, scale, patch_size):
        '''
        预测前的准备
        :param src_img: 输入的切片图像imageCone
        :param scale: 提取图块的倍镜
        :param patch_size: 图块大小
        :return:
        '''
        self._imgCone = src_img
        self.scale = scale
        self.patch_size = patch_size

    def predict_input_fn(self, seeds, batch_size):
        '''
        在predict时，数据输入
        :param seeds: 图块的中心点的集合
        :param batch_size: 每批处理的数量（Tensor计算用的）
        :return:
        '''
        image_list = []
        for x, y in seeds:
            block= self._imgCone.get_image_block(self.scale, x, y, self.patch_size, self.patch_size)
            image_list.append(block.get_img().tobytes())

        src_images = tf.constant(image_list)

        dataset = tf.data.Dataset.from_tensor_slices(src_images)
        dataset = dataset.map(self._parse_function_predict)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        return dataset

    def _parse_function_predict(self, image_data):
        '''
        在predict时，数据解析
        :param image_data: 图像的Byte流
        :return:图像Tensor
        '''
        image_decoded = tf.decode_raw(image_data, tf.uint8)
        image_float = tf.cast(image_decoded, dtype=tf.float32)
        return {"x":image_float}

    def predict(self, src_img, scale, patch_size, seeds):
        '''
        进行预测
        :param src_img:  输入的切片图像imageCone
        :param scale: 提取图块的倍镜
        :param patch_size:图块的大小
        :param seeds: 提取图块的中心点的坐标集合
        :return:
        '''
        self.prepare_predict(src_img, scale, patch_size)

        GPU = False
        if GPU:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu %的显存
            gpu_config.gpu_options.allow_growth = True  # 程序按需申请内存

            config = tf.estimator.RunConfig(keep_checkpoint_max=3, session_config=gpu_config)
        else:
            config = tf.estimator.RunConfig(keep_checkpoint_max=3)

        model_fn = simpleNet128.simpleNet128_model_fn
        classifier = tf.estimator.Estimator(
            model_fn= model_fn, model_dir=self.model_root, config=config)
        batch_size = 20

        # predictions = classifier.predict(input_fn=lambda:self.predict_input_fn(seeds, batch_size))
        seeds_itor = self.get_seeds_itor(seeds, 100)
        result = []
        for part_seeds in seeds_itor:
            predictions = classifier.predict(input_fn=lambda: self.predict_input_fn(part_seeds, batch_size))
            for pred_dict in predictions:
                class_id = pred_dict['classes']
                probability = pred_dict['probabilities'][class_id]
                # print(class_id, 100 * probability)
                result.append((class_id, probability))

        return result

    def get_seeds_itor(self, seeds, batch_size):
        '''
        将所有的图块中心点集，分割成更小的集合，以便减小图块载入的内存压力
        :param seeds: 所有的需要计算的图块中心点集
        :param batch_size: 每次读入到内存的图块数量（不是Tensor的批处理数量）
        :return: 种子点的迭代器
        '''
        len_seeds = len(seeds)
        start_id = 0
        for end_id in range(batch_size + 1, len_seeds, batch_size):
            new_seeds = seeds[start_id : end_id]
            start_id = end_id
            yield new_seeds

        if start_id < len_seeds:
            yield seeds[start_id:len_seeds]

        return

