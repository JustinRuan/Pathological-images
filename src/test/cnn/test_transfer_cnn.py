#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-30'

"""

import unittest
from core import *
from cnn import *

class Test_transfer_cnn(unittest.TestCase):

    def test_loading(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
        tc = transfer_cnn.transfer_cnn(c, "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt", 'loss3/classifier', "Small")

        glcm_features, cnn_features, tags = tc.loading_data(tc.train_list)
        top100_index = tc.select_features(cnn_features, tags)

        tc.train_svm(glcm_features,cnn_features,tags,top100_index)
        tc.test_svm()
        return

