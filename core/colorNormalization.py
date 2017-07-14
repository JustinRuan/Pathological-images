#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, img_as_float
from skimage.color import rgb2gray, rgb2lab, lab2rgb, lab2xyz, xyz2rgb
from matplotlib import pyplot as plt
import numpy as np
import cv2
import copy

#读取图像
ori = io.imread('E:/PythonProjects/pathologicalImages/data/3950.tif')
#源图像进行灰度变换
binry = rgb2gray(ori)
#提取灰度值大于215的点所对应的Mask矩阵 WhitePixel，以及黑背景nwMask
whitePixel = (binry >= 215/255)
nwMask = (binry < 215/255)
#设定各个通道归一化所用的Std，Mean
meanStdTarget = [[77.5121, 270.5718], [8.9287, -23.6535], [2.9664, 8.3857]]
#meanStdTarget = [[78.282154, 300.371584], [9.694320, -10.856946], [2.081496, 3.614328]]
lStd = meanStdTarget[0][0]
lMT = meanStdTarget[0][1]
aStd = meanStdTarget[1][0]
aMT = meanStdTarget[1][1]
bStd = meanStdTarget[2][0]
bMT = meanStdTarget[2][1]
#RGB to LAB变换
#labO = cv2.cvtColor(ori, cv2.COLOR_RGB2LAB)
labO = rgb2lab(ori)
#LAB三通道分离
labO_l = labO[:, :, 0]
labO_a = labO[:, :, 1]
labO_b = labO[:, :, 2]
#分别计算LAB三个通道的图像的黑背景部分所对应的Std，Mean
lsbO = np.std(labO_l[nwMask])
asbO = np.std(labO_a[nwMask])
bsbO = np.std(labO_b[nwMask])

lMO = np.mean(labO_l[nwMask])
aMO = np.mean(labO_a[nwMask])
bMO = np.mean(labO_b[nwMask])

#按设定的Std，Mean进行归一化整个图像
'''labO[:, :, 0] = ((labO[:, :, 0] - lMO)/lsbO) * lStd + lMT
labO[:, :, 1] = ((labO[:, :, 1] - aMO)/asbO) * aStd + aMT
labO[:, :, 2] = ((labO[:, :, 2] - bMO)/bsbO) * bStd + bMT
labO = np.dstack([labO[:, :, 0], labO[:, :, 1], labO[:, :, 2]])'''
#LAB to RGB变换
#RGB = cv2.cvtColor(ori, cv2.COLOR_LAB2RGB)
RGB = lab2rgb(labO)
#RGB三通道分离
RGB_r = RGB[:, :, 0]
RGB_g = RGB[:, :, 1]
RGB_b = RGB[:, :, 2]
#只对黑背景的部分更新了归一化后的值，其余的保留
ori1 = copy.copy(ori/255)
imgR = ori1[:, :, 0]
imgG = ori1[:, :, 1]
imgB = ori1[:, :, 2]
imgR[nwMask] = RGB_r[nwMask]
imgG[nwMask] = RGB_g[nwMask]
imgB[nwMask] = RGB_b[nwMask]
img_r = imgR
img_g = imgG
img_b = imgB
img = np.dstack([img_r, img_g, img_b])
fig, axes = plt.subplots(2, 3, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(ori)
ax[0].set_title("Original image")

ax[1].imshow(binry, cmap=plt.cm.gray)
ax[1].set_title("Gray image")

ax[2].imshow(whitePixel, cmap=plt.cm.gray)
ax[2].set_title("whitePixel")

ax[3].imshow(nwMask, cmap=plt.cm.gray)
ax[3].set_title("nwmask")

ax[4].imshow(img)
ax[4].set_title("img")

for a in ax.ravel():
    a.axis('off')

plt.show()