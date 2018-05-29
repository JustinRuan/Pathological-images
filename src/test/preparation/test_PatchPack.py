#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import unittest
from core import *
from preparation import *

class TestPatchPack(unittest.TestCase):

    def test_packA(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        result = pack.loading("normalA", "cancerA")

        print(result)

        pack.create_train_test_data(900, 900, 1000, 1000, "ZoneA")

    def test_packR(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        (pos, neg) = pack.loading("normal", "cancer")

        print(pos, neg)

        posTrain = int(pos * 0.6)
        negTrain = int(neg * 0.6)
        posTest = int(pos * 0.2)
        negTest = int(neg * 0.2)
        posCheck = pos - posTrain - posTest
        negCheck = neg - negTrain - negTest

        pack.create_train_test_data(posTrain, negTrain, posTest, negTest, posCheck, negCheck, "ZoneR")

# if __name__ == '__main__':
#         unittest.main()
