#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-07-03'

"""
import tensorflow as tf

# -------------------参数------------------------
width = 256
height = 256
channel = 3
learning_rate = 0.000005
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