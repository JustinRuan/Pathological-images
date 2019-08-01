#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-07-31'

"""
import numpy as np
from skimage.morphology import square, dilation, erosion
from skimage import morphology
from scipy.interpolate import griddata
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans

class CancerMapBuilder(object):
    def __init__(self, params, x1, y1, x2, y2, scale):
        self._params = params
        GLOBAL_SCALE = self._params.GLOBAL_SCALE
        xx1, yy1, xx2, yy2 = np.rint(np.array([x1, y1, x2, y2]) * GLOBAL_SCALE / scale).astype(np.int)
        self.valid_area_width = xx2 - xx1
        self.valid_area_height = yy2 - yy1

    def generating_probability_map(self, history, extract_scale, patch_size):
        seeds_scale = self._params.GLOBAL_SCALE
        amplify = extract_scale / seeds_scale
        selem_size = int(0.5 * patch_size / amplify)

        value = np.array(list(history.values()))
        point = list(history.keys())
        value_softmax = 1 / (1 + np.exp(-value))

        # 生成坐标网格
        grid_y, grid_x = np.mgrid[0: self.valid_area_height: 1, 0: self.valid_area_width: 1]
        cancer_map = griddata(point, value_softmax, (grid_x, grid_y), method='linear', fill_value=0)

        cancer_map = morphology.closing(cancer_map, square(2 * selem_size))
        cancer_map = morphology.dilation(cancer_map, square(selem_size))

        return cancer_map

    def calc_probability_threshold(self, history):
        value = np.array(list(history.values()))
        # value_softmax = 1 / (1 + np.exp(-value))
        value = np.reshape(value, (-1, 1))

        clustering = MiniBatchKMeans(n_clusters=2, init='k-means++', max_iter=100, compute_labels=True,
                                     batch_size=200, tol=1e-4).fit(value)
        cluster_centers = clustering.cluster_centers_.ravel()
        index = np.argmax(cluster_centers)

        right_part = value[index == clustering.labels_.ravel()]
        if cluster_centers[index] > 0:
            low_thresh = np.min(right_part)
        else:
            ift = IsolationForest(behaviour='new', max_samples='auto', contamination=0.001)
            y = ift.fit_predict(right_part)
            outliers = right_part[y == -1]

            low_thresh = np.min(outliers)

        return 1 / (1 + np.exp(-low_thresh))













