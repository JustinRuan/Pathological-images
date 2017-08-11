#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, filters, color, morphology, feature, measure
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from sklearn import preprocessing

from post_process import post_process
from colorNormalization import colorNormalization
from anisodiff2D import anisodiff2D
from core.frst import FastRadialSymmetryTransform

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
img = ori
#颜色归一化
#img = colorNormalization(ori)
H = color.rgb2hed(img)
H = H[:, :, 0]
H = -H
H_diff = anisodiff2D(H, niter=15, gamma=1/7)
#归一化到【0,1】
H_diff = (H_diff - H_diff.min()) / (H_diff.max() - H_diff.min())
Lmatrix = np.zeros(H.shape, dtype=np.uint16)
#进入多尺度计算阶段
radius = [3, 5, 9]
#for r in radius :
#便于调试，先取r = 3
r = 3
SE = morphology.disk(r)
# 首先对H_diff进行腐蚀
morph2 = morphology.erosion(H_diff, selem=SE)
# 形态学重建
morph = morphology.reconstruction(morph2, H_diff)
# 再腐蚀
morph2 = morphology.dilation(morph, SE)
# 对补图像再重建，试图建立腐蚀、膨胀两者之间的、稳定过渡区
morph = 1 - (morphology.reconstruction((1 - morph2), (1 - morph)))
# 模板变小一半
SE = morphology.disk(int(r / 2))
morph2 = morphology.dilation(morph, SE)
morph = 1 - (morphology.reconstruction((1 - morph2), (1 - morph)))
morph[morph > 200 / 255] = 1
# 快速径向变换
frst = FastRadialSymmetryTransform()
ns = np.arange(2, 9, 2)
S = frst.transform(morph, ns, 2, 0.5)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
S = min_max_scaler.fit_transform(S)
# 计算morph图像的Sobel梯度，得到灰度梯度图像Gmag
Gmag = filters.sobel(morph)
#Gmag = ndi.filters.sobel(morph)
#Gmag = filters.rank.gradient(morph, morphology.disk(2)) #计算梯度
# 对图像S进行extended maxima transform，得到BW图像
BW = feature.peak_local_max(S, min_distance=10, indices=False)
#BW = morphology.h_maxima(S, 102)
#BW = morphology.local_maxima(S, SE)
# 对BW进行距离变换，得到D
D = ndi.distance_transform_edt(BW)
#寻找峰值
local_maxi = feature.peak_local_max(D, min_distance=10, indices=False)
#初始标记点
markers = ndi.label(local_maxi)[0]
#基于距离变换的分水岭算法
DL = morphology.watershed(-D, markers, mask=BW)
# 提取分水岭所记的背景部分bak
bak = (DL == 0)
# 对BW进行腐蚀，并更新BW
BW = morphology.dilation(BW, SE)
# 将Gmag中BW（细胞核中心部分）和bak（非细胞核部分）所对应部分设成背景
Gmag[BW] = 0
Gmag[bak] = 0
#Gmag[BW] = float('-inf')
#Gmag[bak] = float('-inf')
#基于梯度变换的分水岭算法
L = morphology.watershed(Gmag, markers, mask=bak)
# 对分水岭所标记的区域进行过滤，选择满足要求的区域
# 计算各个标记区域的Area、Solidity，几何中心与重心的距离，椭圆离心率
STATS = measure.regionprops(L)
regfrst, frstLL, Lfrst, STATSfrst = post_process(H, L, r, STATS)

    # 进行多尺度的叠加
    #frstLL[frstLL != 0] = frstLL[frstLL != 0] + (Lmatrix.max()).max()
    #Lmatrix = Lmatrix + frstLL

fig, axes = plt.subplots(2, 3, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(morph, cmap=plt.cm.gray)
ax[0].set_title("morph")

ax[1].imshow(S, cmap=plt.cm.gray)
ax[1].set_title("S")

ax[2].imshow(BW, cmap=plt.cm.gray)
ax[2].set_title("BW")

ax[3].imshow(Gmag, cmap=plt.cm.gray)
ax[3].set_title("Gmag")

ax[4].imshow(L)
ax[4].set_title("L")

ax[5].imshow(frstLL, cmap=plt.cm.gray)
ax[5].set_title("frstLL")

for a in ax.ravel():
    a.axis('off')

plt.show()