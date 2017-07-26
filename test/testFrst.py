#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, color
import numpy as np
from matplotlib import pyplot as plt

from frst import FastRadialSymmetryTransform

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
grayImg= color.rgb2gray(ori)

frst = FastRadialSymmetryTransform()
ns = np.arange(2,9,2)
# ns = [12]
frstImg = frst.transform(grayImg, ns, 2, 0.5)
from scipy.misc import imsave
imsave('frst.jpg', frstImg * 255)
fig, axes = plt.subplots(2, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(ori)
ax[0].set_title("Original image")

ax[1].imshow(grayImg, cmap=plt.cm.gray)
ax[1].set_title("grayImage")

ax[1].imshow(frstImg, cmap=plt.cm.gray)
ax[1].set_title("frstimg")

for a in ax.ravel():
    a.axis('off')

plt.show()



