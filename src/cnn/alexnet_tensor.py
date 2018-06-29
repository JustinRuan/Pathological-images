#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-26'

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from sklearn import metrics
from skimage import io, util
from core import *

tf.logging.set_verbosity(tf.logging.INFO)

# -------------------参数------------------------
width = 256
height = 256
channel = 3
learning_rate = 0.000005
# n_epoch = 70  # 所有训练集数据训练n_epoch代
# train_batch_size = 2
# val_batch_size = 2
num_classes = 2
# -------------------参数------------------------

def alexnet_model_fn(features, labels, mode):
    # -----------------构建网络----------------------
    # # 占位符，类似于变量，在计算图开始计算之前，它们可以用feed_dict将值传入对应的占位符。
    # x = tf.placeholder(tf.float32, shape=[None, width, height, channel], name='x')
    # y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')

    input_layer = tf.reshape(features["x"], [-1, width, height, channel])

    # ********AlexNet有八层网络结构********

    # 第一个卷积层
    # 标准差 stddev，越小幅度越大
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        strides=4,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0))

    # LRN 局部响应归一化
    lsize1 = 4
    lrn1 = tf.nn.lrn(conv1, lsize1, name='norm1')

    # 池化层
    pool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=3, strides=2, padding='same')

    # 第二个卷积层
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0))

    # LRN 局部响应归一化
    lsize2 = 4
    lrn2 = tf.nn.lrn(conv2, lsize2, name='norm2')

    # 池化层
    pool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2, padding='same')

    # 第三个卷积层
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0))

    # 第四个卷积层
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0))

    # 第五个卷积层
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0))

    # 池化层
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2, padding='same')

    # 将图像拉长
    re1 = tf.reshape(pool5, [-1, 8 * 8 * 256])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=4096,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.1))

    dropout_dense1 = tf.nn.dropout(dense1, keep_prob=0.5)

    dense2 = tf.layers.dense(inputs=dropout_dense1,
                             units=4096,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.1))

    # 最后一层
    W_fc1 = tf.Variable(tf.truncated_normal([4096, num_classes], stddev=0.1))  # weight：正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_classes]))  # bias：创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    softmax = tf.nn.softmax(tf.matmul(dense2, W_fc1) + b_fc1, name="softmax_tensor")
    # ---------------------------网络结束-------------num_classes--------------

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=softmax, axis=1),
        # `logging_hook`.
        "probabilities": softmax
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]))  # 交叉熵（损失值）
    # labels = tf.cast(labels, tf.int64)
    temp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=softmax)
    loss = tf.reduce_mean(temp, name="loss")

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


class alexnet_tensor(object):

    # samples_name = "ZoneR"
    def __init__(self, params, samples_name):
        '''
        初始化CNN分类器
        :param params: 参数
        :param model_name: 使用的模型文件
        :param samples_name: 使用的标本集的关键字（标记符）
        '''
        model_name = "alexnet_tensorflow"
        self._params = params
        self.model_name = model_name

        self.model_root = "{}/models/{}/".format(self._params.PROJECT_ROOT, model_name)
        self.train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.check_list = "{}/{}_check.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        # self.CSV_COLUMN_NAMES = ['x', 'label']
        # self.CSV_TYPES = [[""], [0.0]]

        return

    # def _parse_line(self, line):
    #     # Decode the line into its fields
    #     fields = tf.decode_csv(line, record_defaults=self.CSV_TYPES)
    #
    #     # Pack the result into a dictionary
    #     features = dict(zip(self.CSV_COLUMN_NAMES, fields))
    #
    #     # Separate the label from the features
    #     label = features.pop('label')
    #
    #     path = tf.constant(self._params.PATCHS_ROOT_PATH + '/')
    #     patch_file = tf.string_join([path, features['x']])
    #
    #     image_string = tf.read_file(patch_file)
    #     image_decoded = tf.image.decode_image(image_string)
    #     image_float = tf.cast(image_decoded, dtype=tf.float32)
    #     return {"x":image_float}, label
    #
    # def csv_input_fn(self, csv_path, batch_size):
    #     # Create a dataset containing the text lines.
    #     dataset = tf.data.TextLineDataset(csv_path)
    #
    #     # Parse each line.
    #     dataset = dataset.map(self._parse_line)
    #
    #     # Shuffle, repeat, and batch the examples.
    #     # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #     dataset = dataset.batch(batch_size)
    #
    #     # Return the dataset.
    #     return dataset

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_float = tf.cast(image_decoded, dtype=tf.float32)
        return {"x":image_float}, label

    # def csv_input_fn(self, csv_path, batch_size):
    #     filenames_list = []
    #     labels_list = []
    #
    #     f = open(csv_path, "r")
    #     lines = f.readlines()
    #     for line in lines:
    #         items = line.split(" ")
    #
    #         tag = int(items[1])
    #         labels_list.append(tag)
    #
    #         patch_file = "{}/{}".format(self._params.PATCHS_ROOT_PATH, items[0])
    #         filenames_list.append(patch_file)
    #
    #     # A vector of filenames.
    #     filenames = tf.constant(filenames_list)
    #
    #     # `labels[i]` is the label for the image in `filenames[i].
    #     labels = tf.constant(labels_list)
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #     dataset = dataset.map(self._parse_function)
    #     dataset = dataset.repeat().prefetch(batch_size)
    #     dataset = dataset.batch(batch_size)
    #
    #     return dataset

    def read_csv_file(self, csv_path):
        filenames_list = []
        labels_list = []

        f = open(csv_path, "r")
        lines = f.readlines()
        for line in lines:
            items = line.split(" ")

            tag = int(items[1])
            labels_list.append(tag)

            patch_file = "{}/{}".format(self._params.PATCHS_ROOT_PATH, items[0])
            filenames_list.append(patch_file)
        return filenames_list, labels_list

    def eval_input_fn(self, csv_path, batch_size):
        filenames_list, labels_list = self.read_csv_file(csv_path)

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
        filenames_list, labels_list = self.read_csv_file(csv_path)

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

            config = tf.estimator.RunConfig(keep_checkpoint_max=1, session_config=gpu_config)
        else:
            config = tf.estimator.RunConfig(keep_checkpoint_max=1)

        # Create the Estimator
        classifier = tf.estimator.Estimator(
            model_fn=alexnet_model_fn, model_dir=self.model_root, config=config)

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

    cnn = alexnet_tensor(c, "Small")
    cnn.training()