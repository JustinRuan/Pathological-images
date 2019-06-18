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
            self.rand_gen_1 = Halton(1)

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

    # def generate_random_by_mask(self, n, x1, x2, y1, y2, mask):
    #     if mask is None:
    #         return self.generate_random(n, x1, x2, y1, y2)
    #
    #     # coordinate_scale和seed_scale是相同的，是GLOBAL_SCALE
    #     if self.mode == "halton":
    #         result_x = []
    #         result_y = []
    #         while len(result_x) < n:
    #             xy = self.rand_gen.get(1)[0]
    #             x = np.rint(xy[0] * (x2 - x1) - 1).astype(np.int32)
    #             y = np.rint(xy[1] * (y2 - y1) - 1).astype(np.int32)
    #             if mask[y, x]:
    #                 result_x.append(x + x1)
    #                 result_y.append(y + y1)
    #
    #     return np.array(result_x), np.array(result_y)

    def generate_random_by_mask(self, n, x1, x2, y1, y2, mask):
        if mask is None:
            return self.generate_random(n, x1, x2, y1, y2)

        # coordinate_scale和seed_scale是相同的，是GLOBAL_SCALE
        if self.mode == "halton":

            pos = np.nonzero(mask)
            # space = 4
            # y = (np.rint(pos[0] / space) * space)  # row
            # x = (np.rint(pos[1] / space) * space)  # col
            #
            # result = set()
            # for xx, yy in zip(x, y):
            #     result.add((xx, yy))
            #
            # result = np.array(list(result)).astype(np.int32)
            # x = result[:,0]
            # y = result[:,1]

            y = np.array(pos[0]).astype(np.int32)
            x = np.array(pos[1]).astype(np.int32)

            count = len(x)
            if count > n:
                index = np.rint(np.array(self.rand_gen_1.get(n)) * (count - 1)).astype(np.int32).flatten()
                x = x[index] + x1
                y = y[index] + y1
            else:
                x = x + x1
                y = y + y1

        return x, y
