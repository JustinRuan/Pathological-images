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
        result = util.latest_checkpoint("D:/CloudSpace/WorkSpace/PatholImage/models/simplenet128_W1")
        print(result)

