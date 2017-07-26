#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import io, filters, color, morphology, feature, measure
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from post_process import post_process
from colorNormalization import colorNormalization
from anisodiff2D import anisodiff2D
from frst import FastRadialSymmetryTransform

ori = io.imread('E:/PythonProjects/Pathological-images/data/3950.tif')
img = ori
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
for r in radius :
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
    morph = 1 - (morphology.reconstruction(1 - morph2, 1 - morph))
    morph[morph > 200 / 255] = 1
    # 快速径向变换
    frst = FastRadialSymmetryTransform()
    # frstImage = np.array(morph)
    ns = np.arange(2,9,2)
    S = frst.transform(morph, ns, 2, 0.5)
    #S = frst.transform(morph, 2, 12)
    # 计算morph图像的Sobel梯度，得到灰度梯度图像Gmag
    Gmag = filters.sobel(morph)
    # 对图像S进行extended maxima transform，得到BW图像
    BW = feature.peak_local_max(S, min_distance=10, indices=False)
    # BW = morphology.h_maxima(S, 0.4)
    # BW = morphology.local_maxima(S)
    # 对BW进行距离变换，得到D
    D = ndi.distance_transform_edt(BW)
    markers = filters.rank.gradient(D, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = filters.sobel(D)
    # gradient = filters.rank.gradient(D, morphology.disk(2))  # 计算梯度
    DL = morphology.watershed(gradient, markers, mask=D)  # 基于梯度的分水岭算法
    # 提取分水岭所记的背景部分bak
    bak = (DL == 0)
    # 对BW进行腐蚀，并更新BW
    BW = morphology.dilation(BW, SE)
    # 将Gmag中BW（细胞核中心部分）和bak（非细胞核部分）所对应部分设成背景
    Gmag[BW] = 0
    Gmag[bak] = 0
    # Gmag如果是【0,1】分水岭无法得到L？？？？？
    Gmag = Gmag * 255
    # 对更新后的Gmag进行分水岭分割，得到L（准边缘区域）
    distance = ndi.distance_transform_edt(Gmag)
    local_maxi1 = feature.peak_local_max(distance, min_distance=10, indices=False)
    markers1 = ndi.label(local_maxi1)[0]
    L = morphology.watershed(-distance, markers1, mask=Gmag)
    # 对分水岭所标记的区域进行过滤，选择满足要求的区域
    # 计算各个标记区域的Area、Solidity，几何中心与重心的距离，椭圆离心率
    STATS = measure.regionprops(L)
    regfrst, frstLL, Lfrst, STATSfrst = post_process(H, L, r, STATS)

    # 进行多尺度的叠加
    frstLL[frstLL != 0] = frstLL[frstLL != 0] + (Lmatrix.max()).max()
    Lmatrix = Lmatrix + frstLL

fig, axes = plt.subplots(2, 3, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(Gmag,cmap=plt.cm.gray)
ax[0].set_title("Gmag")

ax[1].imshow(Lmatrix, cmap=plt.cm.gray)
ax[1].set_title("Lmatrix")

'''ax[2].imshow(markers)
ax[2].set_title("markers")

ax[3].imshow(L)
ax[3].set_title("L")

ax[4].imshow(BW)
ax[4].set_title("BW")

ax[5].imshow(Lmatrix)
ax[5].set_title("Lmatrix")'''

for a in ax.ravel():
    a.axis('off')

plt.show()