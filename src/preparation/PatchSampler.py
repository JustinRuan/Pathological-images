#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-23'

"""

import numpy as np
from skimage import color, morphology
from skimage.morphology import square
import os
from preparation import PatchFeature
from feature import FeatureExtractor
from core.util import get_seeds

class PatchSampler(object):
    def __init__(self, params):
        self._params= params
        self.seeds_NA = ()
        self.seeds_NR = ()
        self.seeds_TA = ()
        self.seeds_TR = ()
        self.extract_scale = 0

    # # MaksLow 低分辨率Mask图像，种子点在高分辨率图像之间的间隔spacingHigh
    # def get_seeds(self, MaskLow, lowScale, highScale, spacingHigh):
    #     amp = highScale / lowScale
    #     patch_size = int(self._params.PATCH_SIZE_HIGH / amp) # patch size low
    #
    #     # 灰度图像腐蚀，图像中物体会收缩/细化：https://wenku.baidu.com/view/c600c8d1360cba1aa811da73.html
    #     seed_img = morphology.binary_erosion(MaskLow, square(patch_size))
    #     seed_img = morphology.binary_erosion(seed_img, square(8))  # 留边
    #
    #     space_patch = spacingHigh / amp
    #     pos = seed_img.nonzero()
    #     y = (np.rint(pos[0] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # row
    #     x = (np.rint(pos[1] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # col
    #
    #     resultHigh = set()
    #     for xx, yy in zip(x, y):
    #         resultHigh.add((xx, yy))
    #
    #     return resultHigh

    def generate_seeds4_high(self, sourceCone, lowScale, highScale ):
        self.extract_scale = highScale

        mask1 = sourceCone.create_mask_image(lowScale, "TA")
        mask2 = sourceCone.create_mask_image(lowScale, "TR")
        mask3 = sourceCone.create_mask_image(lowScale, "NA")
        mask4 = sourceCone.create_mask_image(lowScale, "NR")
        mask5 = sourceCone.get_effective_zone(lowScale)
        mask4 = mask4 & mask5

        # highScale = self._params.EXTRACT_SCALE
        # lowScale = self._params.GLOBAL_SCALE
        patch_spacing = self._params.PATCH_SIZE_HIGH / 2
        patch_size = self._params.PATCH_SIZE_HIGH

        self.seeds_TA = get_seeds(mask1, lowScale, highScale,patch_size, patch_spacing)
        tmpSeeds = get_seeds(mask2, lowScale, highScale, patch_size, patch_spacing)
        self.seeds_TR = tmpSeeds.difference(self.seeds_TA)

        self.seeds_NA = get_seeds(mask3, lowScale, highScale, patch_size, patch_spacing)

        tmpSeeds2 = get_seeds(mask4, lowScale, highScale,patch_size, patch_spacing)
        self.seeds_NR = tmpSeeds2.difference(self.seeds_NA)

        return (len(self.seeds_TA), len(self.seeds_TR), len(self.seeds_NA), len(self.seeds_NR))

    def detect_patch_byMask(self, mask_img, x, y, patch_width_High, lowScale, highScale):
        amp = highScale / lowScale
        patch_width = patch_width_High / amp
        xlow = int(x / amp)
        ylow = int(y / amp)
        half = int(patch_width / 2)

        sub_m = mask_img[ylow - half: ylow + half, xlow - half: xlow + half]
        total = sub_m.sum()
        r = total / (patch_width * patch_width)
        return r > 0.85

    def extract_patches_AZone(self, sourceCone, scale):
        if (scale != self.extract_scale):
            print("\a", "scale error!")
            return

        Root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(self.extract_scale * 100).astype(np.int)

        pathCancer = "{}/S{}_{}".format(Root_path,intScale, "cancerA")
        pathNormal = "{}/S{}_{}".format(Root_path,intScale, "normalA")
        # pathUnsure = "{}/S{}_{}".format(Root_path,intScale, "unsure")

        if (not os.path.exists(pathCancer)):
            os.makedirs(pathCancer)

        if (not os.path.exists(pathNormal)):
            os.makedirs(pathNormal)

        # if (not os.path.exists(pathUnsure)):
        #     os.makedirs(pathUnsure)

        patch_size = self._params.PATCH_SIZE_HIGH

        for (x, y) in self.seeds_TA:
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            block.save_img(pathCancer)

        for (x, y) in self.seeds_NA:
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            block.save_img(pathNormal)

        return

    def extract_patches_RZone(self, sourceCone, scale):
        if (scale != self.extract_scale):
            print("\a", "scale error!")
            return

        Root_path = self._params.PATCHS_ROOT_PATH
        intScale = np.rint(self.extract_scale * 100).astype(np.int)

        pathCancer = "{}/S{}_{}".format(Root_path,intScale, "cancerR")
        pathNormal = "{}/S{}_{}".format(Root_path,intScale, "normalR")
        pathUnsure = "{}/S{}_{}".format(Root_path,intScale, "unsure")
        pathFalseCancer = "{}/S{}_{}".format(Root_path,intScale, "falseCancerR")
        pathFalseNormal = "{}/S{}_{}".format(Root_path, intScale, "falseNormalR")

        if (not os.path.exists(pathCancer)):
            os.makedirs(pathCancer)

        if (not os.path.exists(pathNormal)):
            os.makedirs(pathNormal)

        if (not os.path.exists(pathUnsure)):
            os.makedirs(pathUnsure)

        if (not os.path.exists(pathFalseCancer)):
            os.makedirs(pathFalseCancer)

        if (not os.path.exists(pathFalseNormal)):
            os.makedirs(pathFalseNormal)

        patch_size = self._params.PATCH_SIZE_HIGH

        pf = PatchFeature.PatchFeature(self._params)
        classifier = pf.load_svm_model()
        extractor = FeatureExtractor.FeatureExtractor()

        lenTR = len(self.seeds_TR)
        for idx, (x, y) in enumerate(self.seeds_TR):
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            img = block.get_img()
            feature = extractor.extract_glcm_feature(img)
            predicted = classifier.predict_proba([feature])
            tag = self.detect_patch_byProb(predicted[0])
            if (tag == 1):
                block.save_img(pathCancer)
            elif (tag == 0):
                print("{} / {} Find Normal patch in TR Zone, {}".format(idx, lenTR, block.encoding()))
                block.save_img(pathFalseCancer)
            else:
                print("{} / {} Find Unsure patch in TR Zone, {} {}".format(idx, lenTR, block.encoding(), predicted[0]))
                block.save_img(pathUnsure)

            if (idx % 1000 == 0):
                print("TR -> Processing: {} / {}".format(idx, lenTR))

        lenNR = len(self.seeds_NR)
        for idx, (x, y) in enumerate(self.seeds_NR):
            block = sourceCone.get_image_block(self.extract_scale, x, y, patch_size, patch_size)
            img = block.get_img()
            feature = extractor.extract_glcm_feature(img)
            predicted = classifier.predict_proba([feature])
            tag = self.detect_patch_byProb(predicted[0])
            if (tag == 1):
                print("{} / {} Find Cancer patch in NR Zone, {}".format(idx, lenNR, block.encoding()))
                block.save_img(pathFalseNormal)
            elif (tag == 0):
                block.save_img(pathNormal)
            else:
                print("{} / {} Find Unsure patch in NR Zone, {} {}".format(idx,  lenNR, block.encoding(), predicted[0]))
                block.save_img(pathUnsure)

            if (idx % 1000 == 0):
                print("NR -> Processing: {} / {}".format(idx, lenNR))

        return

    def detect_patch_byProb(self, probs):
        normal_prob = probs[0]
        cancer_prob = probs[1]
        if (normal_prob > 0.90):
            return 0
        elif (cancer_prob > 0.90):
            return 1
        else:
            return -1