#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-06-19'

"""

import os
import unittest
from core import *
from pytorch.elastic_classifier import Elastic_Classifier

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"

class TestElastic_Classifier(unittest.TestCase):
    def test_evaluate_model_based_slice(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "e_densenet_22"
        sample_name = "4000_256"

        cnn = Elastic_Classifier(c, model_name, sample_name)
        cnn.evaluate_model_based_slice(samples_name=("P0430", "T1_P0430_4000_256_train.txt"),
                                       batch_size=20, max_count=None, slice_count = 0)
