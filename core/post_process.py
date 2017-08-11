#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
import numpy as np
import math
from border_saliency import border_saliency
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
       #WC = STATS[j].weighted_centroid   #这里无法计算重心
       #WCx = WC[0]
       #WCy = WC[1]
       #d = np.hypot(Cx - WCx, Cy - WCy)
       EC = STATS[j].eccentricity
       #按条件进行过滤，面积不在【r，4r】圆面积之内，Solidity过小，d过小，EC太椭了（圆是0），这些区域置成背景
       if A < (r**2) * math.pi or A > ((4 * r)**2) * math.pi or S < 0.815 or EC > 0.9 :
           #labels为分水岭所标记的区域，置0
           labels[labels == j] = 0
       else :
           #当前标记所在区域
           mask = labels == j
           regions[nreg] = j
           newL[mask] = nreg
           nreg += 1

           #输入为区域标记，原图像
           '''l = border_saliency(mask, Hdeconv)
           #如果两种边缘的灰度差值小于20，则该区域不在边缘处
           if l < 1:
               labels[labels == j] = 0
           else:
           #记录最终找到的标记区域newL
               regions[nreg] = j
               newL[mask] = nreg
               nreg+=1'''

    #对newL进行椭圆拟合，得到LL（主程序中的frstLL）
    STATout = regionprops(newL)
    LL = np.zeros(newL.shape)
    for i in range(len(STATout)):
        if STATout[i].major_axis_length != 0 and STATout[i].minor_axis_length != 0:
            rr, cc = ellipse(STATout[i].centroid[0], STATout[i].centroid[1],
                             STATout[i].minor_axis_length, STATout[i].major_axis_length,
                             rotation=STATout[i].orientation)
            if max(rr) < LL.shape[0] and max(cc) < LL.shape[1]:
                LL[rr, cc] = 1

    return regions, LL, newL, STATout