#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-24'

"""
import  os
from feature import FeatureExtractor
from skimage import io

class PatchFeature(object):
    def __init__(self, params):
        self._params = params
        return

    # train_filename = "ZoneA_train.txt"
    def loading_data(self, train_filename):
        root_path = self._params.PATCHS_ROOT_PATH
        train_file = "{}/{}".format(root_path, train_filename)

        fe = FeatureExtractor.FeatureExtractor()
        features = []
        tags = []
        f = open(train_file, "r")
        for line in f:
            items = line.split(" ")
            patch_file = "{}/{}".format(root_path, items[0])
            img = io.imread(patch_file, as_grey=True)
            tag = int(items[1])
            fvector = fe.extract_glcm_feature(img)

            features.append(fvector)
            tags.append(tag)

        f.close()

        return features, tags