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
        features, tags = pf.loading_data("A3_5x64_train.txt")
        print(len(features))

        pf.train_svm(features, tags)

    def test_testSVM(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pf = PatchFeature.PatchFeature(c)
        result = pf.test_svm("A3_5x64_test.txt")
        print(result)

if __name__ == '__main__':
        unittest.main()