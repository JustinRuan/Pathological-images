#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-06-02'

"""
import numpy as np
from skimage import color, morphology
from skimage.morphology import square

# MaksLow 低分辨率Mask图像，种子点在高分辨率图像之间的间隔spacingHigh
def get_seeds(MaskLow, lowScale, highScale, patch_size_high, spacingHigh, margin = -8):
    amp = highScale / lowScale
    patch_size = int(patch_size_high / amp)  # patch size low

    if margin < 0:
        # 灰度图像腐蚀，图像中物体会收缩/细化：https://wenku.baidu.com/view/c600c8d1360cba1aa811da73.html
        seed_img = morphology.binary_erosion(MaskLow, square(patch_size))
        seed_img = morphology.binary_erosion(seed_img, square(margin))  # 收缩边界
    elif margin > 0:
        seed_img = morphology.binary_dilation(MaskLow, square(patch_size))
        seed_img = morphology.binary_dilation(seed_img, square(margin))  # 扩展边界

    space_patch = spacingHigh / amp
    pos = seed_img.nonzero()
    y = (np.rint(pos[0] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # row
    x = (np.rint(pos[1] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # col

    resultHigh = set()
    for xx, yy in zip(x, y):
        resultHigh.add((xx, yy))

    return resultHigh