#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-07-03'

"""
import tensorflow as tf

width = 256
height = 256
channel = 3
learning_rate = 0.000005

def inception(x, p1, p2, p3, p4, scope):
    p1f11 = p1
    p2f11, p2f33 = p2
    p3f11, p3f55 = p3
    p4f11 = p4
    with tf.variable_scope(scope):
        path1 = tf.layers.conv2d(x, filters=p1f11, kernel_size=1, activation=tf.nn.relu, name='p1f11')

        path2 = tf.layers.conv2d(x, p2f11, 1, activation=tf.nn.relu, name='p2f11')
        path2 = tf.layers.conv2d(path2, p2f33, 3, padding='same', activation=tf.nn.relu, name='p2f33')

        path3 = tf.layers.conv2d(x, p3f11, 1, activation=tf.nn.relu, name='p3f11')
        path3 = tf.layers.conv2d(path3, p3f55, 5, padding='same', activation=tf.nn.relu, name='p3f55')

        path4 = tf.layers.max_pooling2d(x, pool_size=3, strides=1, padding='same', name='p4p33')
        path4 = tf.layers.conv2d(path4, p4f11, 1, activation=tf.nn.relu, name='p4f11')

        out = tf.concat((path1, path2, path3, path4), axis=-1, name='path_cat')
    return out

def googlenet_model_fn(features, labels, mode):
    input = tf.reshape(features["x"], [-1, width, height, channel])

    with tf.variable_scope('GoogLeNet'):
        net = tf.layers.conv2d(                 # [batch, 28, 28, 1]
            inputs=input,
            filters=12,
            kernel_size=5,
            strides=1,
            padding='same',
            name="conv1")                       # -> [batch, 28, 28, 12]
        net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool1")                   # -> [batch, 14, 14, 12]
        net = inception(net, p1=64, p2=(6, 64), p3=(6, 32), p4=32, scope='incpt1')  # -> [batch, 14, 14, 64+64+32+32=192]
        net = tf.layers.max_pooling2d(net, 3, 2, padding='same', name="maxpool1")   # -> [batch, 7, 7, 192]
        net = inception(net, p1=256, p2=(32, 256), p3=(32, 128), p4=128, scope='incpt2')  # -> [batch, 7, 7, 768]
        net = tf.layers.average_pooling2d(net, 7, 1, name="avgpool")                # -> [batch, 1, 1, 768]
        net = tf.layers.flatten(net, name='flat')                                   # -> [batch, 768]
        logits = tf.layers.dense(net, 10, name='fc4')                               # -> [batch, n_classes]

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # `logging_hook`.
        "probabilities": logits
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = tf.cast(labels, tf.int64)
    temp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(temp, name="loss")
    # loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]))  # 交叉熵（损失值）

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
    return