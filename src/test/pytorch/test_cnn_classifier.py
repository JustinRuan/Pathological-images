#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import unittest
import numpy as np
from core import *
from pytorch.cnn_classifier import CNN_Classifier
import torch
from preparation.normalization import HistNormalization
from preparation.augmentation import ImageAugmentation

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "cifar10"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name=None, batch_size=32, epochs = 200)

    def test_train_model_patholImg(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        # model_name = "densenet_22"
        # sample_name = "500_128"
        sample_name = "4000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        # augment = ImageAugmentation(l_range = (0.95, 1.05), a_range = (0.95, 1.05),
        #                            b_range = (0.95, 1.05), constant_range = (-10, 10))
        # cnn.train_model(samples_name=("P0327","T_NC_Simple0327_2_{}".format(sample_name)), augment_func = augment,
        #                 batch_size=30, epochs = 3)
        cnn.train_model(samples_name=("P0327", "Aug_LAB_4000_256"), augment_func=None,
                        batch_size=30, epochs = 10)


    def test_evaluate_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "se_densenet_22"
        # sample_name = "x_256"
        model_name = "simple_cnn"
        sample_name = "4000_256"

        # normal = HistNormalization("match_hist", hist_target ="hist_templates.npy",
        #                            hist_source = None)
        # normal = HistNormalization("match_hist", hist_target = "hist_templates.npy",
        #                             hist_source = "hist_soures.npy")
        normal = None

        cnn = CNN_Classifier(c, model_name, sample_name, normalization=normal)
        cnn.evaluate_model(samples_name=("P0330", "T_NC_Simple0330_{}".format(sample_name)),
                           model_file=None, batch_size=20, max_count=None)
        # cnn.evaluate_model(samples_name="T_NC_Simple0327_2_{}".format(sample_name), model_file=None, batch_size=20)
        # cnn.evaluate_model(samples_name=("P0404", "T_NC_Simple0404_4000_256"),
        #                    model_file=None, batch_size=20, max_count=None)
        # cnn.evaluate_model(samples_name=("P0327", "Aug_LAB_4000_256"),
        #                    model_file=None, batch_size=20, max_count=None)

    def test_evaluate_model_with_sampling(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        sample_name = "4000_256"

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Testing/images/test_001.tif",
                                 None, "test_001")
        normal = HistNormalization.get_normalization_function(imgCone, c, 40, 256)
        cnn = CNN_Classifier(c, model_name, sample_name, normalization=normal)
        cnn.evaluate_model(samples_name=("P0404", "T_NC_P0404_{}".format(sample_name)),
                           model_file=None, batch_size=10)

    def test_predict_on_batch(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        sample_name = "4000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(69400, 98400), (70800, 98400), (27200, 113600)] # C, C, S,
        result = cnn.predict_on_batch(imgCone, 40, 256, seeds, 1)
        print(result)

    def test_Image_Dataset(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        # sample_name = "500_128"
        sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        train_data, test_data = cnn.load_custom_data("T_NC_{}".format(sample_name))
        print(train_data.__len__())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
        print(train_loader)

        for index, (x, y) in enumerate(train_loader):
            print(x.shape, y)
            if index > 10: break

    def test_train_model_multi_task(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_22"
        sample_name = "x_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model_multi_task(samples_name="T_NC_{}".format(sample_name), batch_size=32, epochs = 30)

    def test_evaluate_model_multi_task(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_22"
        sample_name = "x_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.evaluate_model_multi_task(samples_name="T_NC_{}".format(sample_name), batch_size=32)

    def test_predict_multi_scale(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_22"
        sample_name = "x_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
        # def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
        result = cnn.predict_multi_scale(imgCone, (10, 20, 40), 256, 20, seeds, 4)
        print(result)

    # def test_01(self):
    #     c_set = np.array([1,1,0])
    #     p_set = np.array([0.91, 0.81, 0.89])
    #     results = []
    #     belief1 = c_set == 1
    #     belief0 = c_set == 0
    #     max_1 = np.max(p_set[belief1])
    #     max_0 = np.max(p_set[belief0])
    #     if max_1 > max_0:
    #         results.append((1, max_1))
    #     else:
    #         results.append((0, max_0))
    #
    #     print(results)

    def test_02(self):
        a = np.random.rand(5,3)
        print(a)
        b = np.max(a, axis=1)
        print(b)

    def test_export_ONNX_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_22"
        # model_name = "simple_cnn"
        sample_name = "x_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        cnn.export_ONNX_model()
        # cnn.export_tensorboard_model()

    def test_train_model_msc(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_c9_22"
        sample_name = "msc_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        samples_name = {10:"T_NC2_msc_256_S1000", 20:"T_NC2_msc_256_S2000", 40:"T_NC2_msc_256_S4000"}
        cnn.train_model_msc(samples_name=samples_name, batch_size=10, epochs = 30)

    def test_Image_Dataset_MSC(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_c9_22"
        sample_name = "msc_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        samples_name = {10: "T_NC_msc_256_S1000", 20: "T_NC_msc_256_S2000", 40: "T_NC_msc_256_S4000"}
        train_data, test_data = cnn.load_msc_data(samples_name)
        print(train_data.__len__())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, num_workers=2)
        print(train_loader)

        for index, (x, y) in enumerate(train_loader):
            print(x.shape, y)
            if index > 10: break

    def test_evaluate_model_msc(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_c9_22"
        sample_name = "msc_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        samples_name = {10: "T_NC_msc_256_S1000", 20: "T_NC_msc_256_S2000", 40: "T_NC_msc_256_S4000"}
        cnn.evaluate_model_msc(samples_name, batch_size=32)

    def test_predict_msc(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "se_densenet_c9_22"
        sample_name = "msc_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
        # def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
        result = cnn.predict_msc(imgCone, 256, 20, seeds, 4)
        print(result)