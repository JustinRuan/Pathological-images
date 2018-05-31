#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-30'

"""

import unittest
from core import *
from cnn import *
import pandas as pd
import numpy as np

class Test_transfer_cnn(unittest.TestCase):

    def test_loading(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
        # tc = transfer_cnn.transfer_cnn(c, "my_googlenet.caffemodel",
        #                                "my_googlenet_deploy.prototxt", 'loss3/classifier', "ZoneA")
        tc = transfer_cnn.transfer_cnn(c, "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt", 'loss3/classifier')

        glcm_features, cnn_features, tags = tc.loading_data("Small500_train.txt")
        top_index = tc.select_features(cnn_features, tags)

        tc.train_svm(glcm_features,cnn_features,tags,top_index)
        tc.test_svm("Small500_train.txt")
        return

    def test_testSVM(self):
        c = Params.Params()
        c.load_config_file("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
        tc = transfer_cnn.transfer_cnn(c, "bvlc_googlenet.caffemodel",
                                       "deploy_GoogLeNet.prototxt", 'loss3/classifier')
        # tc = transfer_cnn.transfer_cnn(c, "my_googlenet.caffemodel",
        #                                "my_googlenet_deploy.prototxt", 'loss3/classifier', "ZoneA")
        tc.test_svm("Small500_train.txt")
        return

    def test_01(self):
        tags = [1, 0, 1, 0]
        cnn_features = [[1,1,1],
                        [0,1,0],
                        [0,1,3],
                        [1,2,1]]
        data = np.column_stack((tags, cnn_features))
        df = pd.DataFrame(data)

        diff = df.groupby(0).mean()
        print(diff)
        absDiff = abs(diff.iloc[0,:] - diff.iloc[1,:]).sort_values(ascending=False)
        print(abs(diff.iloc[0,:] - diff.iloc[1,:]))
        print(absDiff)
        #
        top100_index = absDiff[:2].index.values - 1 # 将index 从1开始移到0开始
        print(top100_index,'\n')
        print(np.array(cnn_features)[:,top100_index])
        return

