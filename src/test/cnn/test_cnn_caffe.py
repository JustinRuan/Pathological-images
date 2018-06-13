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

        cnn = cnn_classifier.cnn_classifier(c, "googlenet_caffe", "ZoneR")
        cnn.write_net()
        cnn.gen_solver(cnn.solver_proto, cnn.train_proto, cnn.train_proto)

        return

    def test_training(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_classifier.cnn_classifier(c, "googlenet_caffe", "ZoneR")
        cnn.training(cnn.solver_proto)

    def test_testing(self):
        # D:\CloudSpace\DoingNow\WorkSpace\PatholImage\models\googlenet_caffe\googlenet_caffe_iter_7200.caffemodel
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")

        cnn = cnn_classifier.cnn_classifier(c, "googlenet_caffe", "ZoneR")
        cnn.testing("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/models/googlenet_caffe/googlenet_caffe_iter_7200.caffemodel")

        return


if __name__ == '__main__':
    unittest.main()