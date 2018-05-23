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

    def test_pack(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        pack = PatchPack.PatchPack(c)
        result = pack.loading("normalA", "cancerA")

        print(result)

        pack.create_train_test_data(300, 300, "ZoneA")

if __name__ == '__main__':
        unittest.main()
