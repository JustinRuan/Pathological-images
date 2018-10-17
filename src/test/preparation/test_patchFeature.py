#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-24'

"""

import unittest
from core import *
from preparation import *

class TestPatchFeature(unittest.TestCase):

    def test_feature(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pf = PatchFeature.PatchFeature(c)
        features, tags = pf.loading_data("SL_5x128_train.txt")
        print(len(features))

        pf.train_svm(features, tags, 128, "SL")

    def test_testSVM(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pf = PatchFeature.PatchFeature(c)
        result = pf.test_svm("SL_5x128_test.txt", 128, "SL")
        print(result)

if __name__ == '__main__':
        unittest.main()