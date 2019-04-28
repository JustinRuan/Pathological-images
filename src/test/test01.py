#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-19'

"""

import tensorflow as tf


def test():
    a = tf.Variable([1])
    print_op = tf.print(a)
    for i in range(10):
        with tf.control_dependencies([print_op]):
            # a_print = tf.Print(a, ['a_value: ', a])
            a = a + 1
            # print_op = tf.print(a)
    return a

def test():
    a = tf.Variable([1])
    print_op = tf.print(a)
    for i in range(10):
        with tf.control_dependencies([print_op]):
            # a_print = tf.Print(a, ['a_value: ', a])
            a = a + 1
            # print_op = tf.print(a)
    return a


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(test())