#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-17'

"""

import unittest
import numpy as np
from core import Params, ImageCone, Open_Slide
from pytorch.encoder_factory import EncoderFactory

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"


class Test_encoder(unittest.TestCase):

    def test_train_model(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "cifar10"

        ae = EncoderFactory(c, model_name, sample_name, 32)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_train_model2(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "vcae"
        sample_name = "cifar10"

        ae = EncoderFactory(c, model_name, sample_name)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_train_model3(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "ccae"
        sample_name = "cifar10"

        ae = EncoderFactory(c, model_name, sample_name)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_train_model4(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "scae"
        sample_name = "AE_500_32"

        ae = EncoderFactory(c, model_name, sample_name, z_dim= 300)
        ae.train_ae(batch_size=64, epochs = 50)

    def test_train_model5(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "aae"
        sample_name = "AE_500_32"

        ae = EncoderFactory(c, model_name, sample_name, z_dim= 64)
        ae.train_adversarial_ae(batch_size=64, epochs = 50)

    def test_extract_feature(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "cifar10"

        ae = EncoderFactory(c, model_name, sample_name)
        ae.extract_feature(None)

    def test_eval_latent_weight(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "AE_500_32"

        ae = EncoderFactory(c, model_name, sample_name, 64)
        result = ae.eval_latent_vector_loss(batch_size=64)
        np.save("latent_vector_loss_{}".format(model_name), result)

    def test_calc_latent_vector_weight(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "AE_500_32"

        ae = EncoderFactory(c, model_name, sample_name, 16)
        loss = np.load("latent_vector_loss_{}.npy".format(model_name))

        print(loss)

    def test_01(self):
        import torch
        x = torch.zeros(4, 5)

        rnd = torch.randn(4).unsqueeze(1)
        print("rnd, ", rnd)
        index = np.full((4,1), 1)
        # x.scatter_(1, torch.tensor([[1], [1], [1], [1]]), rnd)
        x.scatter_(1, torch.from_numpy(index).long(), rnd)
        print(x)

    def test_02(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "cae"
        sample_name = "AE_500_32"

        ae = EncoderFactory(c, model_name, sample_name, 16)
        model = ae.create_initial_model()
        from torchsummary import summary
        summary(model, input_size=(3, 32, 32), device="cpu")

    def test_03(self):
        from visdom import Visdom
        viz = Visdom()

        # 单张
        viz.image(
            np.random.rand(3, 512, 256),
            opts=dict(title='Random!', caption='How random.'),
        )
        # 多张
        viz.images(
            np.random.randn(20, 3, 64, 64),
            opts=dict(title='Random images', caption='How random')
        )

    def test_train_idec(self):
        c = Params()
        c.load_config_file(JSON_PATH)

        model_name = "idec"
        sample_name = "AE_500_32"

        idec = EncoderFactory(c, model_name, sample_name, 64)
        idec.train_idec(batch_size=64, epochs=100)

