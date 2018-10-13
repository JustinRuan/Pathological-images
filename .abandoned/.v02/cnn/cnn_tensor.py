#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-29'

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from sklearn import metrics
from skimage import io, util
from core import *
from core.util import read_csv_file
from cnn.net import alexnet, googlenet

tf.logging.set_verbosity(tf.logging.INFO)


class cnn_tensor(object):

    # samples_name = "ZoneR"
    def __init__(self, params, model_name, samples_name):
        '''
        初始化CNN分类器
        :param params: 参数
        :param model_name: 使用的模型文件
        :param samples_name: 使用的标本集的关键字（标记符）
        '''
        model_name = model_name + "_tensorflow"
        self._params = params
        self.model_name = model_name

        self.model_root = "{}/models/{}/".format(self._params.PROJECT_ROOT, model_name)
        self.train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.check_list = "{}/{}_check.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        return

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_float = tf.cast(image_decoded, dtype=tf.float32)
        return {"x":image_float}, label

    def eval_input_fn(self, csv_path, batch_size):
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
        batch_size = 10
        classifier.train(
            input_fn=lambda:self.train_input_fn(self.train_list, batch_size, 3),
            steps=30,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = classifier.evaluate(input_fn=lambda:self.eval_input_fn(self.test_list, batch_size))
        print(eval_results)

        return


if __name__ == '__main__':
    c = Params.Params()
    c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

    # cnn = cnn_tensor(c,"alexnet", "Small")
    cnn = cnn_tensor(c, "googlenet", "Small")
    cnn.training()





