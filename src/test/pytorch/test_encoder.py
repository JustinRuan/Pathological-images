#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-17'

"""

import unittest
from core import Params, ImageCone, Open_Slide
from pytorch.encoder import Encoder

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"


class Test_encoder(unittest.TestCase):

    def test_train_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "cifar10"

        ae = Encoder(c, model_name, sample_name)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_train_model2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "vae"
        sample_name = "cifar10"

        ae = Encoder(c, model_name, sample_name)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_extract_feature(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "cifar10"

        ae = Encoder(c, model_name, sample_name)
        ae.extract_feature(None)