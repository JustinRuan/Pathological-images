#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-01-17'

"""

import numpy as np
from core.sobol_gen import SobolGenerator
from ghalton import Halton

class Random_Gen(object):
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "random":
            pass
        elif self.mode == "sobol":
            self.rand_gen = SobolGenerator(2)
        elif self.mode == "halton":
            self.rand_gen = Halton(2)

    def generate_random(self, n, x1, x2, y1, y2):
        if self.mode == "random":
            x = np.random.randint(x1, x2 - 1, size=n, dtype='int')
            y = np.random.randint(y1, y2 - 1, size=n, dtype='int')
        elif self.mode == "sobol":
            xy = self.rand_gen.generate(n)
            x = np.rint(xy[:,0] * (x2 - x1) + x1 - 1).astype(np.int32)
            y = np.rint(xy[:,1] * (y2 - y1) + y1 - 1).astype(np.int32)
        elif self.mode == "halton":
            xy = np.array(self.rand_gen.get(n))
            x = np.rint(xy[:,0] * (x2 - x1) + x1 - 1).astype(np.int32)
            y = np.rint(xy[:,1] * (y2 - y1) + y1 - 1).astype(np.int32)

        return x, y
