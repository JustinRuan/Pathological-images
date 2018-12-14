#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-14'

"""

import unittest
from core import Params
from pytorch.feature_extractor import Feature_Extractor

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

SAMPLE_FIlENAME = "T_NC_500_128"
# SAMPLE_FIlENAME = "T_NC_2000_256"
# SAMPLE_FIlENAME = "T_NC_4000_256"

# PATCH_TYPE = "4000_256"
# PATCH_TYPE = "2000_256"
PATCH_TYPE = "500_128"


# MODEL_NAME = "inception_v3"
MODEL_NAME = "densenet121"

class Test_feature_extractor(unittest.TestCase):

    def test_extract_features_save_to_file(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        fe = Feature_Extractor(c, MODEL_NAME, PATCH_TYPE)
        fe.extract_features_save_to_file(SAMPLE_FIlENAME, batch_size=32)

