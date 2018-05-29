#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-27'

"""

from core import *
from preparation import PatchFeature, PatchPack, PatchSampler

class SamplePreparation(object):

    def open_config(self,config_filename):
        self._params = Params.Params()
        self._params.load_config_file(config_filename)

    def open_slide(self, filename, ano_filename, id_string):
        self.imgCone = ImageCone.ImageCone(self._params)

        # 读取数字全扫描切片图像
        tag = self.imgCone.open_slide(filename, ano_filename, id_string)
        return tag

    def extract_patches_AZone(self):
        ps = PatchSampler.PatchSampler(self._params)
        highScale = self._params.EXTRACT_SCALE
        lowScale = self._params.GLOBAL_SCALE

        result = ps.generate_seeds4_high(self.imgCone, lowScale, highScale)
        print(result)

        ps.extract_patches_AZone(self.imgCone, highScale)

    def create_train_data_zoneA(self):
        pack = PatchPack.PatchPack(self._params)
        result = pack.loading("normalA", "cancerA")

        print(result)
        pack.create_train_test_data(900, 900, 1000, 1000, "ZoneA")

    def train_svm_zoneA(self):
        pf = PatchFeature.PatchFeature(self._params)
        features, tags = pf.loading_data("ZoneA_train.txt")
        print(len(features))
        pf.train_svm(features, tags)

    def extract_patches_RZone(self):
        ps = PatchSampler.PatchSampler(self._params)
        highScale = self._params.EXTRACT_SCALE
        lowScale = self._params.GLOBAL_SCALE

        result = ps.generate_seeds4_high(self.imgCone, lowScale, highScale)
        print(result)

        ps.extract_patches_RZone(self.imgCone, highScale)

    def create_train_data_zoneR(self):
        pack = PatchPack.PatchPack(self._params)
        (pos, neg) = pack.loading("normal", "cancer")

        print(pos, neg)

        posTrain = int(pos * 0.7)
        negTrain = int(neg * 0.7)

        pack.create_train_test_data(posTrain, negTrain, pos - posTrain, neg - negTrain, "ZoneR")


if __name__ == '__main__':
    sp = SamplePreparation()
    sp.open_config("D:/CloudSpace/DoingNow/WorkSpace/PatholImage/config/justin.json")
    tag = sp.open_slide("17004930 HE_2017-07-29 09_45_09.kfb",
                                 '17004930 HE_2017-07-29 09_45_09.kfb.Ano', "17004930")

    if tag:
        sp.extract_patches_AZone()
        sp.create_train_data_zoneA()
        sp.train_svm_zoneA()
        sp.extract_patches_RZone()
        sp.create_train_data_zoneR()