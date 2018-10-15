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

    def test_pack_5x256(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        data_tag = []
        result = pack.loading_filename_tags("S500_256_cancer", 1)
        data_tag.extend(result)
        result = pack.loading_filename_tags("S500_256_normal", 0)
        data_tag.extend(result)

        print(len(data_tag))

        pack.create_train_test_data(data_tag, 0.7, 0.3, "test_5x256")

    # def test_packR(self):
    #     c = Params.Params()
    #     c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
    #
    #     pack = PatchPack.PatchPack(c)
    #     (pos, neg) = pack.loading("normal", "cancer")
    #
    #     print(pos, neg)
    #
    #     posTrain = int(pos * 0.6)
    #     negTrain = int(neg * 0.6)
    #     posTest = int(pos * 0.2)
    #     negTest = int(neg * 0.2)
    #     posCheck = pos - posTrain - posTest
    #     negCheck = neg - negTrain - negTest
    #
    #     pack.create_train_test_data(posTrain, negTrain, posTest, negTest, posCheck, negCheck, "ZoneR")

# if __name__ == '__main__':
#         unittest.main()
