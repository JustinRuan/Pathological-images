#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import numpy as np
from skimage import color, morphology
from skimage.morphology import square

class PatchSampler(object):
    def __init__(self, params):
        self._params= params
        self.seeds_NA = ()
        self.seeds_NR = ()
        self.seeds_TA = ()
        self.seeds_TR = ()

    # MaksLow 低分辨率Mask图像，种子点在高分辨率图像之间的间隔spacingHigh
    def get_seeds(self, MaskLow, lowScale, highScale, spacingHigh):
        amp = highScale / lowScale
        patch_size = int(self._params.PATCH_SIZE_HIGH / amp) # patch size low

        # 灰度图像腐蚀，图像中物体会收缩/细化：https://wenku.baidu.com/view/c600c8d1360cba1aa811da73.html
        seed_img = morphology.binary_erosion(MaskLow, square(patch_size))
        seed_img = morphology.binary_erosion(seed_img, square(8))  # 留边

        space_patch = spacingHigh / amp
        pos = seed_img.nonzero()
        y = (np.rint(pos[0] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # row
        x = (np.rint(pos[1] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # col

        resultHigh = set()
        for xx, yy in zip(x, y):
            resultHigh.add((xx, yy))

        return resultHigh

    def generate_seeds4_high(self, sourceCone, lowScale, HighScale):
        mask1 = sourceCone.create_mask_image(lowScale, "TA")
        mask2 = sourceCone.create_mask_image(lowScale, "TR")
        mask3 = sourceCone.create_mask_image(lowScale, "NA")
        mask4 = sourceCone.create_mask_image(lowScale, "NR")
        mask5 = sourceCone.get_roi(lowScale)
        mask4 = mask4 & mask5

        # highScale = self._params.EXTRACT_SCALE
        # lowScale = self._params.GLOBAL_SCALE
        patch_spacing = self._params.PATCH_SIZE_HIGH / 2

        self.seeds_TA = self.get_seeds(mask1, lowScale, HighScale, patch_spacing)
        tmpSeeds = self.get_seeds(mask2, lowScale, HighScale, patch_spacing)
        self.seeds_TR = tmpSeeds.difference(self.seeds_TA)

        self.seeds_NA = self.get_seeds(mask3, lowScale, HighScale, patch_spacing)

        tmpSeeds2 = self.get_seeds(mask4, lowScale, HighScale, patch_spacing)
        self.seeds_NR = tmpSeeds2.difference(self.seeds_NA)

        return (len(self.seeds_TA), len(self.seeds_TR), len(self.seeds_NA), len(self.seeds_NR))



