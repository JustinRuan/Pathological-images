#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import unittest
import numpy as np
from core import *
from skimage import io
from pytorch.cnn_classifier import Simple_Classifier, DMC_Classifier
import torch
from preparation.normalization import HistNormalization, HSDNormalization, ACDNormalization,ReinhardNormalization #, ACDNormalization_tf
from preparation.augmentation import ImageAugmentation, RndAugmentation, HRndAugmentation
from pytorch.image_dataset import Image_Dataset
import pandas as pd
from pytorch.elastic_classifier import Elastic_Classifier

import warnings
warnings.filterwarnings('ignore')

# JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class Test_cnn_classifier(unittest.TestCase):

    def test_train_model_cifar(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "e_densenet_40"
        sample_name = "cifar10"

        cnn = Simple_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name=None, augment_func=None, batch_size=32, epochs = 200)

    def test_train_model_patholImg(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        mode = 41

        if mode == 1:
            model_name = "simple_cnn"
            sample_name = "4000_256"
        elif mode == 2:
            model_name = "densenet_22"
            sample_name = "4000_256"
        elif mode == 3:
            model_name = "e_densenet_22"
            sample_name = "4000_256"
        elif mode == 4:
            model_name = "e_densenet_40"
            sample_name = "4000_256"
        elif mode == 41:
            model_name = "e_densenet_40"
            sample_name = "2000_256"
        elif mode == 5:
            model_name = "se_densenet_40"
            sample_name = "4000_256"
        elif mode == 6:
            model_name = "resnet_18"
            sample_name = "4000_256"
        elif mode == 7:
            pass

        cnn = Simple_Classifier(c, model_name, sample_name)
        samples = [("P0619","T1_P0619_4000_256"),  # 平衡样本集
                   ("P0619","S1_P0619_2000_256"), # 平衡样本集
                   ("P0619","A1_P0619_4k2k_256"), # 平衡样本集
        ]

        # cnn.train_model(samples_name=("P0430","T1_P0430_4000_256"), class_weight=[0.2391, 1.0],
        #                 augment_func = None, batch_size=30, epochs = 10)
        # cnn.train_model(samples_name=samples[1], cl ass_weight=None,
        #                 augment_func = None, batch_size=30, epochs = 10)
        cnn.train_model_A2(samples_name=samples[1], augment_func = None, class_weight=None,
                        batch_size=30, loss_weight=0.001, epochs = 10)


    def test_evaluate_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "se_densenet_40"
        # sample_name = "x_256"
        # model_name = "simple_cnn"
        model_name = "e_densenet_22"
        sample_name = "4000_256"

        # normal = HistNormalization("match_hist", hist_target ="hist_templates_2048.npy",
        #                            hist_source = "hist_source_P0404.npy")
        # normal = HistNormalization("match_hist", hist_target = "hist_soures_P0327.npy",
        #                             hist_source = "hist_soures_P0330.npy")
        # normal = HSDNormalization("hsd_norm", target_mean=( -0.2574, 0.2353, 0.3893),
        #                           target_std=(0.1860, 0.1884, 0.2482),
        #                           source_mean=(-0.0676, 0.4088, 0.3710),
        #                           source_std=(0.1254, 0.1247, 0.1988))
        # normal = ReinhardNormalization("reinhard", target_mean=(72.66, 16.89, -9.979),
        #                             target_std=(13.42, 5.767, 4.891),
        #                             source_mean=(72.45, 17.63, -17.77),
        #                             source_std=(10.77, 7.064, 6.50))

        normal = None
        # normal = ImageAugmentation(l_range = (0.95, 1.05), a_range = (0.95, 1.05),
        #                            b_range = (0.95, 1.05), constant_range = (-10, 10))

        # normal = ACDNormalization_tf("acd", dc_txt="dc.txt", w_txt="w.txt", template_path="template_normal")

        # normal = ACDNormalization("acd", dc_txt="dc.txt", w_txt="w.txt", template_path="Tumor_025")
        # source_samples = ("P0404", "T_NC_Simple0404_4000_256_test.txt")
        # patch_root = c.PATCHS_ROOT_PATH[source_samples[0]]
        # sample_filename = source_samples[1]
        # train_list = "{}/{}".format(patch_root, sample_filename)
        #
        # Xtrain, Ytrain = util.read_csv_file(patch_root, train_list)
        #
        # # prepare
        # images = []
        # for patch_file in Xtrain:
        #     img = io.imread(patch_file, as_gray=False)
        #     images.append(img)
        #
        # normal.prepare(images)

        cnn = Simple_Classifier(c, model_name, sample_name, normalization=normal, special_norm= -1)
        # cnn.evaluate_model(samples_name=("P0330", "T_NC_Simple0330_4000_256_test.txt"),
        #                    model_file=None, batch_size=20, max_count=None)

        # cnn.evaluate_model(samples_name=("P0404", "T_NC_Y0404_4000_256_test.txt"),
        #                    model_file=None, batch_size=20, max_count=None)

        # cnn.evaluate_model(samples_name=("P0404", "T_NC_W0404_4000_256_test.txt"),
        #                    model_file=None, batch_size=20, max_count=None)

        # cnn.evaluate_model(samples_name=("P0330", "T_NC_Simple0330_4000_256_test.txt"),
        #                    model_file=None, batch_size=60, max_count=None)

        cnn.evaluate_model(samples_name=("P0404", "T_NC_Simple0404_4000_256_test.txt"), # T_NC_Simple0404_4000_256_test
                           model_file=None,
                           batch_size=20, max_count=None)
        # cnn.evaluate_model(samples_name=("P0430", "Check_P0430_4000_256_test.txt"), # Check_P0430_4000_256_test, T1_P0430_4000_256_train
        #                    model_file=None, batch_size=100, max_count=None )

    def test_evaluate_model2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "se_densenet_40"
        # sample_name = "x_256"
        # model_name = "simple_cnn"
        model_name = "e_densenet_22"
        sample_name = "4000_256"

        normal = None

        cnn = Simple_Classifier(c, model_name, sample_name, normalization=normal, special_norm= -1)

        # Check_P0430_4000_256_test,
        # T1_P0430_4000_256_train,
        # T1_P0430_4000_256_test.txt
        Xtest, Ytest, predicted_tags, features = cnn.evaluate_model(
            samples_name=("P0430", "T1_P0430_4000_256_train.txt"),
            model_file=None, batch_size=100, max_count=None)
        cnn.evaluate_accuracy_based_slice(Xtest, predicted_tags, Ytest)
        # cnn.visualize_features(Xtest, features, Ytest)

        #the center of Label  0  =  [ 1.4759356 -1.5035709]
        #the center of Label  1  =  [-1.5735501  1.5415243]

    def test_evaluate_model3(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "se_densenet_40"
        # model_name = "simple_cnn"
        # model_name = "resnet_18"
        model_name = "densenet_22"
        sample_name = "4000_256"

        cnn = Simple_Classifier(c, model_name, sample_name, normalization=None, special_norm= -1)

        result = cnn.evaluate_models(samples_name=("P0404", "T_NC_Simple0404_4000_256_test.txt"),
                           model_directory="{}/models/pytorch/{}_{}".format(c.PROJECT_ROOT,model_name, sample_name),
                                           batch_size=60, max_count=None)

        pd.set_option('display.max_colwidth', 300)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        print(result)

    def test_evaluate_model_with_sampling(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        sample_name = "4000_256"

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Testing/images/test_001.tif",
                                 None, "test_001")
        images = imgCone.get_uniform_sampling_blocks(40, 256)
        print("sample count = ", len(images))

        normal = ACDNormalization("acd", dc_txt="dc.txt", w_txt="w.txt", template_path="template_normal")
        normal.prepare(images)

        source_samples = ("P0404", "T_NC_Simple0404_4000_256_test.txt")
        cnn = Simple_Classifier(c, model_name, sample_name, normalization=normal, special_norm= 0)
        cnn.evaluate_model(samples_name=source_samples,
                           model_file=None,
                           batch_size=20, max_count=None)

    def test_predict_on_batch(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        sample_name = "4000_256"

        cnn = Simple_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(69400, 98400), (70800, 98400), (27200, 113600)] # C, C, S,
        result = cnn.predict_on_batch(imgCone, 40, 256, seeds, 2)
        print(result)

    def test_predict_on_batch_Elastic(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet_22"
        sample_name = "4000_256"

        cnn = Elastic_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Train_Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(69400, 98400), (70800, 98400), (27200, 113600)]  # C, C, S,
        result = cnn.predict_on_batch(imgCone, 40, 256, seeds, 2)
        print(result)


    def test_Image_Dataset(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "simple_cnn"
        # model_name = "densenet_22"
        # sample_name = "500_128"
        sample_name = "4000_256"

        patch_root = c.PATCHS_ROOT_PATH["P0327"]
        sample_filename = "T_NC_Simple0327_2_{}".format(sample_name)
        train_list = "{}/{}_train.txt".format(patch_root, sample_filename)

        Xtrain, Ytrain = util.read_csv_file(patch_root, train_list)
        Xtrain, Ytrain = Xtrain[:40], Ytrain[:40]  # for debug

        train_data = Image_Dataset(Xtrain, Ytrain, )  # transform = None,

        print(train_data.__len__())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, num_workers=2)
        print(train_loader)

        for index, (x, y) in enumerate(train_loader):
            print(x.shape, y)
            if index > 10: break

    def test_export_ONNX_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "se_densenet_22"
        model_name = "e_densenet_22"
        sample_name = "4000_256"

        cnn = Simple_Classifier(c, model_name, sample_name)

        cnn.export_ONNX_model()
        # cnn.export_tensorboard_model()

    def test_DSC_train_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "dsc_densenet_40"
        sample_name = "2040_256"

        samples = [("P0619","T1_P0619_4000_2000_256"),  # 平衡样本集
                    ("P0619","T2_P0619_4000_2000_256"),]
        cnn = DMC_Classifier(c, model_name, sample_name)

        cnn.train_model(samples_name=samples[1], class_weight=None,
                        batch_size=20, epochs = 10)
        # cnn.train_model_A2(samples_name=samples[0], class_weight=None,
        #                 batch_size=20, loss_weight=0.001, epochs = 5)
        # cnn.train_model_A3(samples_name=samples[1], class_weight=None,
        #                 batch_size=20, loss_weight=0.001, epochs = 5)

    def test_speed(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "dsc_densenet_40"
        sample_name = "2040_256"

        samples = [("P0619","T1_P0619_4000_2000_256"),  # 平衡样本集
                    ]
        cnn = DMC_Classifier(c, model_name, sample_name)
        cnn.evaluate_model(samples_name=samples[0], model_file=None, batch_size=100)
        # i5-7500, GTX 1060 6G
        # acc, acc20, acc40, speed，len: 0.9723953695458593 0.9702034385916843 0.9520515103774231 235.46774193548387 14599

        # i7-8700，GTX 1080 8G
        # acc, acc20, acc40, speed: 0.9723953695458593 0.9702034385916843 0.9520515103774231 286.2549019607843

    def test_class_centers(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "dsc_densenet_40"
        sample_name = "2040_256"

        samples = [("P0619","T1_P0619_4000_2000_256"),  # 平衡样本集
                    ]
        cnn = DMC_Classifier(c, model_name, sample_name)
        cnn.calc_centers(samples_name=samples[0], model_file=None, batch_size=100)

    # center_20 Normal[3.0154288 - 3.374219]
    # center_20 Tumor[-2.7567124 2.4602337]
    # center_40 Normal[2.0857253 - 2.4292686]
    # center_40 Tumor[-2.5575078 2.2696517]


    # def test_train_model_multi_task(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_22"
    #     sample_name = "x_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #     cnn.train_model_multi_task(samples_name="T_NC_{}".format(sample_name), batch_size=32, epochs = 30)
    #
    # def test_evaluate_model_multi_task(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_22"
    #     sample_name = "x_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #     cnn.evaluate_model_multi_task(samples_name="T_NC_{}".format(sample_name), batch_size=32)
    #
    # def test_predict_multi_scale(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_22"
    #     sample_name = "x_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #
    #     imgCone = ImageCone(c, Open_Slide())
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("Tumor/Tumor_004.tif",
    #                              None, "Tumor_004")
    #     seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
    #     # def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
    #     result = cnn.predict_multi_scale(imgCone, (10, 20, 40), 256, 20, seeds, 4)
    #     print(result)
    #
    # # def test_01(self):
    # #     c_set = np.array([1,1,0])
    # #     p_set = np.array([0.91, 0.81, 0.89])
    # #     results = []
    # #     belief1 = c_set == 1
    # #     belief0 = c_set == 0
    # #     max_1 = np.max(p_set[belief1])
    # #     max_0 = np.max(p_set[belief0])
    # #     if max_1 > max_0:
    # #         results.append((1, max_1))
    # #     else:
    # #         results.append((0, max_0))
    # #
    # #     print(results)
    #
    # def test_02(self):
    #     a = np.random.rand(5,3)
    #     print(a)
    #     b = np.max(a, axis=1)
    #     print(b)
    #
    # def test_export_ONNX_model(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_22"
    #     # model_name = "simple_cnn"
    #     sample_name = "x_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #
    #     cnn.export_ONNX_model()
    #     # cnn.export_tensorboard_model()
    #
    # def test_train_model_msc(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_c9_22"
    #     sample_name = "msc_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #     samples_name = {10:"T_NC2_msc_256_S1000", 20:"T_NC2_msc_256_S2000", 40:"T_NC2_msc_256_S4000"}
    #     cnn.train_model_msc(samples_name=samples_name, batch_size=10, epochs = 30)
    #
    # def test_Image_Dataset_MSC(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_c9_22"
    #     sample_name = "msc_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #
    #     samples_name = {10: "T_NC_msc_256_S1000", 20: "T_NC_msc_256_S2000", 40: "T_NC_msc_256_S4000"}
    #     train_data, test_data = cnn.load_msc_data(samples_name)
    #     print(train_data.__len__())
    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, num_workers=2)
    #     print(train_loader)
    #
    #     for index, (x, y) in enumerate(train_loader):
    #         print(x.shape, y)
    #         if index > 10: break
    #
    # def test_evaluate_model_msc(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_c9_22"
    #     sample_name = "msc_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #     samples_name = {10: "T_NC_msc_256_S1000", 20: "T_NC_msc_256_S2000", 40: "T_NC_msc_256_S4000"}
    #     cnn.evaluate_model_msc(samples_name, batch_size=32)
    #
    # def test_predict_msc(self):
    #     c = Params()
    #     c.load_config_file(JSON_PATH)
    #
    #     model_name = "se_densenet_c9_22"
    #     sample_name = "msc_256"
    #
    #     cnn = CNN_Classifier(c, model_name, sample_name)
    #
    #     imgCone = ImageCone(c, Open_Slide())
    #
    #     # 读取数字全扫描切片图像
    #     tag = imgCone.open_slide("Tumor/Tumor_004.tif",
    #                              None, "Tumor_004")
    #     seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
    #     # def predict_multi_scale(self, src_img, scale_tuple, patch_size, seeds_scale, seeds, batch_size):
    #     result = cnn.predict_msc(imgCone, 256, 20, seeds, 4)
    #     print(result)