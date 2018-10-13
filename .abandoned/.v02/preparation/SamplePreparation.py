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
        '''
        读取配置文件
        :param config_filename: 配置文件路径
        :return:
        '''
        self._params = Params.Params()
        self._params.load_config_file(config_filename)

    def open_slide(self, filename, ano_filename, id_string):
        '''
        读取切片文件
        :param filename: 切片文件
        :param ano_filename: 对应的标注文件
        :param id_string: 切片的编号
        :return: 是否成功打开
        '''
        self.imgCone = ImageCone.ImageCone(self._params)

        # 读取数字全扫描切片图像
        tag = self.imgCone.open_slide(filename, ano_filename, id_string)
        return tag

    def extract_patches_AZone(self):
        '''
        在精标区内进行图块提取
        :return:
        '''
        ps = PatchSampler.PatchSampler(self._params)
        highScale = self._params.EXTRACT_SCALE
        lowScale = self._params.GLOBAL_SCALE

        result = ps.generate_seeds4_high(self.imgCone, lowScale, highScale)
        print(result)

        ps.extract_patches_AZone(self.imgCone, highScale)

    def create_train_data_zoneA(self):
        '''
        根据 精标区的图块所在目录，生成训练SVM的标本集的列表
        :return:
        '''
        pack = PatchPack.PatchPack(self._params)
        result = pack.loading("normalA", "cancerA")

        print(result)
        pack.create_train_test_data(900, 900, 1000, 1000, 0, 0, "ZoneA")

    def train_svm_zoneA(self):
        '''
        使用精标区的图块，训练SVM
        :return:
        '''
        pf = PatchFeature.PatchFeature(self._params)
        features, tags = pf.loading_data("ZoneA_train.txt")
        print(len(features))
        pf.train_svm(features, tags)

    def extract_patches_RZone(self):
        '''
        从粗标区，提取图块，并存入对应的目录
        :return:
        '''
        ps = PatchSampler.PatchSampler(self._params)
        highScale = self._params.EXTRACT_SCALE
        lowScale = self._params.GLOBAL_SCALE

        result = ps.generate_seeds4_high(self.imgCone, lowScale, highScale)
        print(result)

        ps.extract_patches_RZone(self.imgCone, highScale)

    def create_train_data_zoneR(self):
        '''
        生成基于精标区和粗标区的图块，生成样本集的文件列表
        :return:
        '''
        pack = PatchPack.PatchPack(self._params)
        (pos, neg) = pack.loading("normal", "cancer")

        print(pos, neg)

        posTrain = int(pos * 0.6)
        negTrain = int(neg * 0.6)
        posTest = int(pos * 0.2)
        negTest = int(neg * 0.2)
        posCheck = pos - posTrain - posTest
        negCheck = neg - negTrain - negTest
        pack.create_train_test_data(posTrain, negTrain, posTest, negTest, posCheck, negCheck, "ZoneR")


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