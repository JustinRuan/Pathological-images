#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-24'

"""

import unittest
from core import util


class TestUtil(unittest.TestCase):

    def test_latest_checkpoint(self):
        result = util.latest_checkpoint("D:/CloudSpace/WorkSpace/PatholImage/models/pytorch/se_densenet_22_x_256")
        print(result)



    def test_clean_checkpoint(self):
        util.clean_checkpoint("D:/CloudSpace/WorkSpace/PatholImage/models/pytorch/scae_AE_500_32_300", best_number=10)

    def test_read_csv_file(self):
        filenames_list, labels_list = util.read_csv_file("D:/Data/Patches/P1113/", "D:/Data/Patches/P1113/T_NC_256_test.txt")
        print(len(filenames_list), len(labels_list[0]))
