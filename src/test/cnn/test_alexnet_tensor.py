#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-27'

"""

import unittest
from core import *
from cnn import *
# import tensorflow as tf

class Test_alexnet_tensor(unittest.TestCase):

    def test_training(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        cnn = alexnet_tensor.alexnet_tensor(c, "Small")
        cnn.training()

#     def test_load_imgs(self):
#         c = Params.Params()
#         c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
#
#         cnn = alexnet_tensor.alexnet_tensor(c, "Small")
#
#         # path = tf.constant(cnn._params.PATCHS_ROOT_PATH + '/')
#         # path2 = tf.constant("S2000_cancerR/17004930_019712_020864_2000_0.jpg")
#         # patch_file = tf.string_join([path, path2])
#         # image_string = tf.read_file(patch_file)
#         # image_decoded = tf.image.decode_image(image_string)
#         # image_float = tf.cast(image_decoded, dtype=tf.float32)
#
#         dataset = tf.data.TextLineDataset(cnn.train_list)
#
#         # Parse each line.
#         dataset = dataset.map(_parse_line)
#         dataset = dataset.batch(1)
#         iterator = dataset.make_initializable_iterator()
#         x, y = iterator.get_next()
#
#
#         sess = tf.Session()
#         sess.run(iterator.initializer)
#         sess.run(x)
#         print(x)
#         sess.close()
#
# def _parse_line(line):
#     CSV_COLUMN_NAMES = ['image_path', 'label']
#     CSV_TYPES = [[""], [0.0]]
#     # Decode the line into its fields
#     fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
#
#     # Pack the result into a dictionary
#     features = dict(zip(CSV_COLUMN_NAMES, fields))
#
#     # Separate the label from the features
#     label = features.pop('label')
#
#     # path = tf.constant('D:/Study/breast/Patches/P0523/')
#     # patch_file = tf.string_join([path, features['image_path']])
#     #
#     # image_string = tf.read_file(patch_file)
#     # image_decoded = tf.image.decode_image(image_string, name="x")
#     # image_float = tf.cast(image_decoded, dtype=tf.float32)
#     # return {"x":image_float}, label
#     return features, label