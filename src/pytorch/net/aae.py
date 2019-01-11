#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-01-04'

"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable


#  Adversarial Autoencoder

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        intermediate_size = 128

        # Encoder
        # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
        self.encoder = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)), # 32 x 32 => 32 x 32
                ("relu1", nn.LeakyReLU(0.2, inplace=True)),
                ("bn_c1", nn.BatchNorm2d(32, affine=False)),
                ("conv2", nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)), # 16 x 16
                ("relu2", nn.LeakyReLU(0.2, inplace=True)),
                ("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)), # 16 x 16
                ("relu3", nn.LeakyReLU(0.2, inplace=True)),
                ("bn_c3", nn.BatchNorm2d(64, affine=False)),
                ("conv4", nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0)), # 8 x 8
                ("relu4", nn.LeakyReLU(0.2, inplace=True)),
                ("conv5", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)), # 8 x 8
                ("relu5", nn.LeakyReLU(0.2, inplace=True)),
                ("bn_c5", nn.BatchNorm2d(128, affine=False)),
                ("conv6", nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)), # 4 x 4
                ("relu6", nn.LeakyReLU(0.2, inplace=True)),
            ])
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, intermediate_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(intermediate_size, z_dim),
            nn.BatchNorm1d(z_dim, affine=False),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        return self.encoder_fc(out)

    def encode(self, x):
        return self.forward(x)


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()

        intermediate_size = 128

        # Decoder
        # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
        self.decoder_fc = nn.Sequential(
            nn.Linear(z_dim, intermediate_size),
            nn.ReLU(True),
            nn.Linear(intermediate_size, 2048),
            nn.ReLU(True),
        )

        self.decoder =  nn.Sequential(
            OrderedDict([
                ("deconv1", nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)), #  4x4 => 8 x 8
                ("relu1",   nn.ReLU()),
                ("bn_c1",   nn.BatchNorm2d(128, affine=False)),
                ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)), # 8 x 8
                ("relu2",   nn.ReLU()),
                ("deconv3", nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)), # 16 x 16
                ("relu3",   nn.ReLU()),
                ("bn_c3",   nn.BatchNorm2d(64, affine=False)),
                ("deconv4", nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)), # 16 x 16
                ("relu4",   nn.ReLU()),
                ("deconv5", nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)), # 32 x 32
                ("relu5",   nn.ReLU()),
                ("bn_c5",   nn.BatchNorm2d(32, affine=False)),
                ("deconv6", nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)), # 32 x3 2
                ("sigmoid", nn.Sigmoid()),
            ])
        )

    def forward(self, z):
        out = self.decoder_fc(z)
        out = out.view(out.size(0), 128, 4, 4)
        out = self.decoder(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()

        intermediate_size = 128

        self.fc = nn.Sequential(
            nn.Linear(z_dim, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.fc(z)
        return out


