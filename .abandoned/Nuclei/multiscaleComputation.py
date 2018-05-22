#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import io, filters, color, morphology, feature, measure

from anisodiff2D import anisodiff2D
from frst import FastRadialSymmetryTransform
from post_process import post_process

ori = io.imread('../data/3950.tif')
img = ori
#颜色归一化
#img = colorNormalization(ori)
H = color.rgb2hed(img)
H = H[:, :, 0]

#归一化到【0,255】
H = 255 * (H - H.min()) / (H.max() - H.min())
H_diff = anisodiff2D(H, niter=15, gamma=1/7, kappa=30)

Lmatrix = np.zeros(H.shape, dtype=np.uint16)
LBW = np.zeros(H.shape, dtype=np.uint8)

#进入多尺度计算阶段
radius = [3, 5, 7]
# radius = [3]
for r in radius :
    SE = morphology.disk(r)
    # 首先对H_diff进行腐蚀
    morph2 = morphology.erosion(H_diff, selem=SE)
    # 形态学重建
    morph = morphology.reconstruction(morph2, H_diff, method='dilation')
    # 再腐蚀
    morph2 = morphology.dilation(morph, SE)

    # 对补图像再重建，试图建立腐蚀、膨胀两者之间的、稳定过渡区
    morph = (morphology.reconstruction((255 - morph2), (255 - morph), method='dilation'))
    morph = morphology.erosion(morph, selem=SE)

    # 模板变小一半
    SE = morphology.disk(int(r / 2))
    morph2 = morphology.dilation(morph, SE)
    morph = (morphology.reconstruction((255 - morph2), (255 - morph)))

    morph[morph > 190] = 255
    # morph = morphology.closing(morph, morphology.disk(r))

    # 快速径向变换
    frst = FastRadialSymmetryTransform()
    ns = np.arange(r, r + 8, 2)
    S = frst.transform(morph, ns, 2, 2)

    S = 255 * (S - S.min()) / (S.max() - S.min())
    S[S < 60] = 0

    # 对图像S进行extended maxima transform，得到BW图像
    # BW = feature.peak_local_max(S, min_distance=5, indices=False)
    BW = morphology.h_maxima(S, 5)
    #BW = morphology.local_maxima(S, SE)

    # 对BW进行距离变换，得到D
    D = ndi.distance_transform_edt(BW)
    #寻找峰值
    local_maxi = feature.peak_local_max(D, min_distance=5, indices=False)

    #初始标记点
    markers = ndi.label(local_maxi)[0]
    #分水岭算法
    DL = morphology.watershed(-D, markers, watershed_line=True)

    # 提取分水岭所记的背景部分bak
    bak = (DL == 0)
    bak = morphology.dilation(bak, morphology.disk(3))

    # 对BW进行膨胀，并更新BW
    BW = morphology.dilation(BW, SE)
    LBW = LBW + BW

    # 计算morph图像的Sobel梯度，得到灰度梯度图像Gmag
    # 这里的处理很敏感，对后面的分水岭影响很大
    # Gmag = filters.sobel(morph) + filters.sobel(H_diff)
    Gmag = filters.sobel(H_diff + filters.sobel(morph))
    Gmag = 255 * (Gmag - Gmag.min()) / (Gmag.max() - Gmag.min())
    # Gmag = feature.canny(morph)
    #Gmag = filters.rank.gradient(morph, morphology.disk(2)) #计算梯度

    # 将Gmag中BW（细胞核中心部分）和bak（非细胞核部分）所对应部分设成背景
    Gmag[LBW>0] = -255
    Gmag[bak] = 0
    #
    #基于梯度变换的分水岭算法
    # L = morphology.watershed(Gmag, markers, mask=~bak)
    L = morphology.watershed(Gmag, markers)
    tag = set(L[bak]).pop()
    zoneOne = (L == 1)
    L[L == tag] = 1
    L[zoneOne] = tag

    # 对分水岭所标记的区域进行过滤，选择满足要求的区域
    # 计算各个标记区域的Area、Solidity，几何中心与重心的距离，椭圆离心率
    STATS = measure.regionprops(L, intensity_image = H)
    regfrst, frstLL, Lfrst, STATSfrst = post_process(H, L, r, STATS)

    # 进行多尺度的叠加
    frstLL[frstLL != 0] = frstLL[frstLL != 0] + (Lmatrix.max()).max()
    Lmatrix = Lmatrix + frstLL

######################################################################################
fig, axes = plt.subplots(2, 3, figsize=(4, 3))
ax = axes.ravel()
# ax[0].imshow(H_diff, cmap=plt.cm.gray)
# ax[0].set_title("H_diff")

# ax[1].imshow(morph, cmap=plt.cm.gray)
# ax[1].set_title("morph")

ax[5].imshow(S, cmap=plt.cm.gray)
ax[5].set_title("S")
#
ax[3].imshow(LBW, cmap=plt.cm.gray)
ax[3].set_title("LBW")
#
ax[4].imshow(img)
SE = morphology.disk(6)
tImg = morphology.dilation(LBW, SE)
ax[4].contour(tImg, [0.5], linewidths=0.5, colors='r')
ax[4].set_title("image")
# #
# ax[5].imshow(color.label2rgb(DL))
# ax[5].set_title("DL")

ax[0].imshow(Gmag, cmap=plt.cm.gray)
ax[0].set_title("Gmag")

ax[1].imshow(color.label2rgb(L))
# ax[1].imshow(L)
ax[1].set_title("L")


ax[2].imshow(img)
ax[2].contour(Lmatrix, [0.5], linewidths=0.5, colors='r')
# ax[2].contour(Lfrst, [0.5], linewidths=0.5, colors='g')
# ax[2].imshow(frstLL, cmap=plt.cm.gray)
ax[2].set_title("Lmatrix")

for a in ax.ravel():
    a.axis('off')

plt.show()