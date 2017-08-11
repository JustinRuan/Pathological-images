#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, color
import numpy as np
from matplotlib import pyplot as plt

from core.frst import FastRadialSymmetryTransform
from read import readSlider, readImage
from scipy import ndimage
from sklearn import preprocessing

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
ihc_hed = color.rgb2hed(ori)
img = ndimage.median_filter(ihc_hed[:, :, 0], size=3)
minValue = np.min(img)
img = img - minValue

frst = FastRadialSymmetryTransform()
ns = np.arange(2,9,2)
# ns = [12]
frstImg = frst.transform(img, ns, 1, 2)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
frstImg = min_max_scaler.fit_transform(frstImg)

fig, axes = plt.subplots(2, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(ori)
ax[0].set_title("Original image")

ax[1].imshow(img, cmap=plt.cm.gray)
ax[1].set_title("img")

ax[2].imshow(frstImg, cmap=plt.cm.gray)
ax[2].set_title("frstimg")

for a in ax.ravel():
    a.axis('off')

plt.show()



