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
        data_tag = pack.initialize_sample_tags_NC(["S500_64_cancer"],["S500_64_normal"])

        print(len(data_tag))
        pack.create_train_test_data(data_tag, 0.4, 0.6, "A3_5x64")

    def test_pack_refine_sample_tags_NC(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        pack.refine_sample_tags_NC(["S500_64_cancer"],["S500_64_normal"])

# if __name__ == '__main__':
#         unittest.main()
