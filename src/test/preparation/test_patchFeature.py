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
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        pf = PatchFeature.PatchFeature(c)
        features, tags = pf.loading_data("ZoneA_train.txt")
        print(len(features))

if __name__ == '__main__':
        unittest.main()