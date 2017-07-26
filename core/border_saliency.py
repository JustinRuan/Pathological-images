#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import measure
import numpy as np

def border_saliency(mask, H) :
    # 得到该区域的补invmask
    invmask = ~mask
    # 求出invmask的边缘Bout，并求出Mask的边缘Bin
    Bout = measure.find_contours(invmask, 0.5)
    Bin = measure.find_contours(mask, 0.5)
    # 求出Bout的补的边缘，外边缘的补的边缘，相当于推理出来的内边缘，以及对面
    Bout = measure.find_contours(~Bout, 0.5)
    # 计算Bout(Bin) = 0;%原先的内边缘置0，这时Bout对应外边缘再外推出的边缘，Bin是内边缘
    Bout[Bin] = 0

    # 提出Bout和Bin两部分对应的灰度值，计算它们的中值之差
    inpx = H[Bin]
    outpx = H[Bout]
    mi = np.median(inpx)
    mo = np.median(outpx)
    l = abs(mi - mo)
    return l