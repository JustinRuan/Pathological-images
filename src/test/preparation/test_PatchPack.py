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

    def test_pack_5(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        # data_tag = pack.initialize_sample_tags_NCL(["S500_64_cancer"],["S500_64_normal"],["S500_64_lymph"])
        data_tag = pack.initialize_sample_tags(["S500_128_cancer"], ["S500_128_stroma"])
        # data_tag = pack.initialize_sample_tags_SCL([], ["S500_128_stroma"], ["S500_128_lymph"])

        print(len(data_tag))
        pack.create_train_test_data(data_tag, 0.8, 0.2, "SLC2_5x128")

    def test_pack_refine_sample_tags_SC(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        pack.refine_sample_tags_SC(["S500_128_cancer"],["S500_128_stroma"])


    def test_extract_refine_sample_SC(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        # pack.extract_refine_sample_SC(5, ["S500_128_cancer"],["S500_128_stroma"])
        pack.extract_refine_sample_LE(5, ["S500_128_lymph"],["S500_128_edge"])
# if __name__ == '__main__':
#         unittest.main()
