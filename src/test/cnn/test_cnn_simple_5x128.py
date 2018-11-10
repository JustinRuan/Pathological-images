#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-10'

"""

import unittest
from core import *
from cnn import cnn_simple_5x128

class Test_cnn_simple_5x128(unittest.TestCase):

    def test_training(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_simple_5x128(c, "simplenet128")
        cnn.train_model("CNN_R_500_128", batch_size = 100, augmentation = (True, False))


