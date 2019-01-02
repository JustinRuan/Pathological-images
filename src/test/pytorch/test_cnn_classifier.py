#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import unittest
from core import Params, ImageCone, Open_Slide
from pytorch.cnn_classifier import CNN_Classifier
import torch

JSON_PATH = "H:\yeguanglu\Pathological-images-newest\config\ygl.json"
# JSON_PATH = "C:/RWork/WorkSpace/PatholImage/config/justin_m.json"
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

        # model_name = "simple_cnn"
        # model_name = "densenet_22"
        # sample_name = "500_128"
        model_name="densenet_22"
        sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name="T_NC_{}".format(sample_name), batch_size=64,epochs = 100)

    def test_train_model_patholImg2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        # model_name = "densenet_22"
        # sample_name = "500_128"
        model_name="densenet_40"
        sample_name = "500_128+2000_256+4000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)
        cnn.train_model(samples_name="T_NC_{}".format(sample_name), batch_size=6,epochs = 100)

    def test_train_model_Multitask(self,Multitask=True):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        # model_name = "densenet_22"
        # sample_name = "500_128"
        model_name="sedensenet_22"
        sample_name = "500_128+2000_256+4000_256"

        cnn = CNN_Classifier(c, model_name, sample_name,Multitask)
        cnn.train_model_MultiTask(samples_name="T_NC_{}".format(sample_name), batch_size=64,epochs = 100)


    def test_predict_on_batch(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "densenet_22"
        sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        tag = imgCone.open_slide("Tumor/Tumor_004.tif",
                                 None, "Tumor_004")
        seeds = [(34816, 48960), (35200, 48640), (12800, 56832)] # C, C, S,
        result = cnn.predict_on_batch(imgCone, 20, 256, seeds, 1)
        print(result)


    #单独对每个倍镜下的数据进行测试
    def test_predict_on_samples(self, MultiTask=False):
        c = Params()
        c.load_config_file(JSON_PATH)
        model_name = "sedensenet_40"
        sample_name = "4000_256"
        cnn = CNN_Classifier(c, model_name, sample_name)
        model_file="E:\PythonProjects\Pathological-images-newest\models\pytorch\densenet_22_500_128+2000_256+4000_256\cp-0018-0.1279-0.9522.h5"
        if MultiTask:
            epoch_loss, epoch_acc, epoch_acc_gain=cnn.predict_on_samples(patch_type=sample_name,batch_size=16, model_file=model_file,MultiTask=MultiTask)
            print(epoch_loss, epoch_acc, epoch_acc_gain)
        else:
            epoch_loss, epoch_acc=cnn.predict_on_samples(patch_type=sample_name,batch_size=16, model_file=model_file,MultiTask=MultiTask)
            print(epoch_loss, epoch_acc)

    def test_Image_Dataset(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        # model_name = "simple_cnn"
        model_name = "densenet_22"
        sample_name = "500_128"
        # sample_name = "2000_256"

        cnn = CNN_Classifier(c, model_name, sample_name)

        train_data, test_data = cnn.load_custom_data("T_NC_{}".format(sample_name))
        print(train_data.__len__())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
        print(train_loader)

        for index, (x, y) in enumerate(train_loader):
            print(x.shape, y)
            if index > 10: break