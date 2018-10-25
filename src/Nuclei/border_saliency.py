#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
from skimage import measure, feature, morphology
import numpy as np

def border_saliency(mask, H) :
    # 得到该区域的补invmask
    invmask = ~mask
    # 求出invmask的边缘Bout，并求出Mask的边缘Bin
    # Bout = invmask - morphology.binary_erosion(invmask,morphology.rectangle(3,3))
    Bout = np.bitwise_xor(invmask, morphology.binary_erosion(invmask, morphology.rectangle(3, 3)))
    # Bin = mask - morphology.binary_erosion(mask,morphology.rectangle(3,3))
    Bin = np.bitwise_xor(mask, morphology.binary_erosion(mask,morphology.rectangle(3,3)))
    invBout = ~Bout
    # Bout = invBout - morphology.binary_erosion(invBout,morphology.rectangle(3,3))
    Bout = np.bitwise_xor(invBout, morphology.binary_erosion(invBout, morphology.rectangle(3, 3)))
    Bout[Bin] = 0

    # 提出Bout和Bin两部分对应的灰度值，计算它们的中值之差
    inpx = H[Bin]
    outpx = H[Bout]
    mi = np.median(inpx)
    mo = np.median(outpx)
    value = abs(mi - mo)
    return value