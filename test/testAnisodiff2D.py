#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, color
from matplotlib import pyplot as plt

from core.colorNormalization import colorNormalization
from core.anisodiff2D import anisodiff2D

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
img = colorNormalization(ori)
H = color.rgb2hed(img)
H = H[:, :, 0]
H = -H
H_diff = anisodiff2D(H, niter=15, gamma=1/7)

fig, axes = plt.subplots(2, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(ori)
ax[0].set_title("ori")

ax[1].imshow(img)
ax[1].set_title("img")

ax[2].imshow(H, cmap=plt.cm.gray)
ax[2].set_title("H")

ax[3].imshow(H_diff, cmap=plt.cm.gray)
ax[3].set_title("H_diff")

for a in ax.ravel():
    a.axis('off')

plt.show()