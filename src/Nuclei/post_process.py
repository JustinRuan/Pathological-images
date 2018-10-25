#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
import numpy as np
import math
from Nuclei import border_saliency
from skimage.measure import regionprops
from skimage.draw import ellipse, ellipse_perimeter
from skimage.draw import ellipse

def post_process(Hdeconv, labels, r, STATS) :
    regions = []
    nreg = 1
    newL = np.zeros(labels.shape, dtype=np.uint16)
    for j in range(len(STATS)) :
       regions.append(0)

       A = STATS[j].area
       S = STATS[j].solidity
       C = STATS[j].centroid
       Cx = C[0]
       Cy = C[1]
       WC = STATS[j].weighted_centroid
       WCx = WC[0]
       WCy = WC[1]
       d = np.hypot(Cx - WCx, Cy - WCy)
       EC = STATS[j].eccentricity

       tag = STATS[j].label

       if tag == 1: continue

       #按条件进行过滤，面积不在【r，4r】圆面积之内，Solidity过小，d过小，EC太椭了（圆是0），这些区域置成背景
       if A < (r**2) * math.pi or A > ((8 * r)**2) * math.pi or S < 0.5 or EC > 0.95 or  d<0.04:
           #labels为分水岭所标记的区域，置0
           labels[labels == tag] = 0
       else :
           #当前标记所在区域
           mask = labels == tag

           # 输入为区域标记，原图像
           t = border_saliency(mask, Hdeconv)
           # t = 100
           # 如果两种边缘的灰度差值小于20，则该区域不在边缘处
           if t > 10:
                # 记录最终找到的标记区域newL
               regions[nreg] = tag
               newL[mask] = nreg
               nreg += 1

    #对newL进行椭圆拟合，得到LL（主程序中的frstLL）
    STATout = regionprops(newL)
    LL = np.zeros(newL.shape)
    for i in range(len(STATout)):
        if STATout[i].major_axis_length != 0 and STATout[i].minor_axis_length != 0:
            rr, cc = ellipse(STATout[i].centroid[0], STATout[i].centroid[1],
                             STATout[i].minor_axis_length/2, STATout[i].major_axis_length/2,
                             rotation=STATout[i].orientation)
            if max(rr) < LL.shape[0] and max(cc) < LL.shape[1]:
                LL[rr, cc] = 1

    return regions, LL, newL, STATout