#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io
from matplotlib import pyplot as plt
from core.colorNormalization import colorNormalization

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
img = colorNormalization(ori)

fig, axes = plt.subplots(1, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(ori)
ax[0].set_title("Original image")

ax[1].imshow(img)
ax[1].set_title("img")

for a in ax.ravel():
    a.axis('off')

plt.show()