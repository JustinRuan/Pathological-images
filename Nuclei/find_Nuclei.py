#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import io, filters, color, morphology, feature, measure

from anisodiff2D import anisodiff2D
from frst import FastRadialSymmetryTransform
from post_process import post_process


def read_image(filename):
    # img = io.imread('../data/3950.tif')
    img = io.imread(filename)

    # 颜色归一化
    # img = colorNormalization(ori)
    H = color.rgb2hed(img)
    H = H[:, :, 0]

    # 归一化到【0,255】
    H = 255 * (H - H.min()) / (H.max() - H.min())
    return img, H


def preprocess_morphology(src, radius):
    SE = morphology.disk(radius)
    # 首先对H_diff进行腐蚀
    morph2 = morphology.erosion(src, selem=SE)
    # 形态学重建
    morph = morphology.reconstruction(morph2, src, method='dilation')
    # 再腐蚀
    morph2 = morphology.dilation(morph, SE)

    # 对补图像再重建，试图建立腐蚀、膨胀两者之间的、稳定过渡区
    morph = (morphology.reconstruction((255 - morph2), (255 - morph), method='dilation'))
    morph = morphology.erosion(morph, selem=SE)

    # 模板变小一半
    SE = morphology.disk(int(radius / 2))
    morph2 = morphology.dilation(morph, SE)
    morph = (morphology.reconstruction((255 - morph2), (255 - morph)))

    morph[morph > 190] = 255
    # morph = morphology.closing(morph, morphology.disk(r))
    return morph


def frst2D(src, radius):
    # 快速径向变换
    frst = FastRadialSymmetryTransform()
    ns = np.arange(radius, radius + 8, 2)
    S = frst.transform(src, ns, 2, 2)

    S = 255 * (S - S.min()) / (S.max() - S.min())
    S[S < 60] = 0
    return S


def find_nucleus_center(src, radius):
    # 对图像S进行extended maxima transform，得到BW图像
    # BW = feature.peak_local_max(S, min_distance=5, indices=False)
    BW = morphology.h_maxima(src, 5)
    # BW = morphology.local_maxima(S, SE)

    # 对BW进行距离变换，得到D
    D = ndi.distance_transform_edt(BW)
    # 寻找峰值
    local_maxi = feature.peak_local_max(D, min_distance=5, indices=False)

    # 初始标记点
    markers = ndi.label(local_maxi)[0]

    # 分水岭算法
    DL = morphology.watershed(-D, markers, watershed_line=True)

    # 提取分水岭所记的背景部分bak
    bak = (DL == 0)
    bak = morphology.dilation(bak, morphology.disk(3))

    # 对BW进行膨胀，并更新BW
    BW = morphology.dilation(BW, morphology.disk(radius))
    return BW, markers, bak


def find_cell_boundaries(src, src_filtered, LBW, bak):
    # 计算morph图像的Sobel梯度，得到灰度梯度图像Gmag
    # 这里的处理很敏感，对后面的分水岭影响很大
    Gmag = filters.sobel(src + filters.sobel(src_filtered))
    Gmag = 255 * (Gmag - Gmag.min()) / (Gmag.max() - Gmag.min())

    # 将Gmag中BW（细胞核中心部分）和bak（非细胞核部分）所对应部分设成背景
    Gmag[LBW > 0] = -255
    Gmag[bak] = 0
    return Gmag


def divide_cell_region(Gmag, markers, bak):
    # 基于梯度变换的分水岭算法
    L = morphology.watershed(Gmag, markers)
    tag = set(L[bak]).pop()
    zoneOne = (L == 1)
    L[L == tag] = 1
    L[zoneOne] = tag
    return L


def main():
    print("Start in %s" % __name__)

    img, H = read_image('../data/3950.tif')
    H_diff = anisodiff2D(H, niter=15, gamma=1 / 7, kappa=30)

    Lmatrix = np.zeros(H.shape, dtype=np.uint16)
    LBW = np.zeros(H.shape, dtype=np.uint8)

    # 进入多尺度计算阶段
    radiusArray = [3, 5, 7]
    # radiusArray = [3]
    for r in radiusArray:
        morph = preprocess_morphology(H_diff, r)

        S = frst2D(morph, r)

        BW, markers, bak = find_nucleus_center(S, r)

        LBW = LBW + BW
        # LBW = BW

        Gmag = find_cell_boundaries(H_diff, morph, LBW, bak)

        L = divide_cell_region(Gmag, markers, bak)

        # 对分水岭所标记的区域进行过滤，选择满足要求的区域
        # 计算各个标记区域的Area、Solidity，几何中心与重心的距离，椭圆离心率
        STATS = measure.regionprops(L, intensity_image=H)
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


if __name__ == '__main__':
    main()
