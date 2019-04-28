#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-19'

"""

import tensorflow as tf


def test():
    a = tf.constant(0)
    for i in range(10):
        print_op = tf.print(a, ['a_value: ', a])
        with tf.control_dependencies([print_op]):
            a = a + 1
    return a


if __name__ == '__main__':
    with tf.Session() as sess:
        out = test()
        sess.run(out)
