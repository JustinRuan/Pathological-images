#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-28'

"""

import unittest
from core import *
from cnn import *


class Test_cnn_caffe(unittest.TestCase):

    def test_write_net(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_caffe.cnn_caffe(c, "googlenet_caffe", "ZoneR")
        cnn.write_net()
        cnn.gen_solver(cnn.solver_proto, cnn.train_list, cnn.test_list)


        return


if __name__ == '__main__':
    unittest.main()