#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-25'

"""


import unittest
from core import Params
from cnn import cnn_tensor

class Test_cnn_tensor(unittest.TestCase):

    def test_training(self):
        c = Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_tensor(c, "simplenet128", "SC_5x128")
        cnn.training()