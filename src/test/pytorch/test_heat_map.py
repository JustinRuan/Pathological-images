#!/usr/bin/env python
# encoding: utf-8
'''
@author: Justin Ruan
@license: 
@contact: ruanjun@whut.edu.cn
@time: 2020-01-10
@desc:
'''

import os
import unittest
from core import *
from pytorch.heat_map import HeatMapBuilder

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestHeatMapBuilder(unittest.TestCase):

    def test_create_train_data(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        hmb = HeatMapBuilder(c, model_name="Slide_FCN")
        hmb.create_train_data(chosen=None)

    def test_train(self):

        c = Params()
        c.load_config_file(JSON_PATH)

        hmb = HeatMapBuilder(c, model_name="Slide_FCN")
        hmb.train(batch_size=1, epochs=1)