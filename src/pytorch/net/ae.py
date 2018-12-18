#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-17'

"""

import torch
import torch.nn as nn


#
# class Autoencoder(nn.Module):
#     def __init__(self, out_dim):
#         super(Autoencoder, self).__init__()
#         intermediate_size = 128
#
#         # Encoder
#         # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
#         self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1) #  32 x 32
#         self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0) # 16 x 16
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 16 x 16
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)
#         self.fc2 = nn.Linear(intermediate_size, out_dim)
#
#         # Decoder
#         self.fc3 = nn.Linear(out_dim, intermediate_size)
#         # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
#         self.fc4 = nn.Linear(intermediate_size, 8192)
#         self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1) # 16x16
#         self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0) # 32x32
#         self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1) # 32x32
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.bn = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm1d(out_dim)
#
#     def encode(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.bn(self.relu(self.conv2(out)))
#         out = self.relu(self.conv3(out))
#         out = self.relu(self.conv4(out))
#         out = out.view(out.size(0), -1)
#         h1 = self.relu(self.fc1(out))
#         return self.bn2(self.fc2(h1))
#
#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         out = self.relu(self.fc4(h3))
#         out = out.view(out.size(0), 32, 16, 16)
#         out = self.relu(self.deconv1(out))
#         out = self.bn(self.relu(self.deconv2(out)))
#         out = self.relu(self.deconv3(out))
#         out = self.sigmoid(self.conv5(out))
#         return out
#
#     def forward(self, x):
#         out = self.encode(x)
#         out = self.decode(out)
#         return out

class Autoencoder(nn.Module):
    def __init__(self, out_dim):
        super(Autoencoder, self).__init__()
        intermediate_size = 128

        # Encoder
        # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) #  32 x 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0) # 16 x 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 16 x 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0) # 8 x 8
        self.conv5= nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 8 x 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0) # 4 x 4
        self.fc1 = nn.Linear(4 * 4 * 128, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, out_dim)

        # Decoder
        self.fc3 = nn.Linear(out_dim, intermediate_size)
        # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
        self.fc4 = nn.Linear(intermediate_size, 2048) # 4x4x128
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0) # 8 x 8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1) # 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0) # 16 x 16
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1) # 16 x 16
        self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0) # 32 x 32
        self.deconv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1) # 32x32

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn32 = nn.BatchNorm2d(32, affine=False)
        self.bn64 = nn.BatchNorm2d(64, affine=False)
        self.bn128 = nn.BatchNorm2d(128, affine=False)
        self.bn_out = nn.BatchNorm1d(out_dim, affine=False)

    def encode(self, x):
        out = self.bn32(self.relu(self.conv1(x)))
        out = self.relu(self.conv2(out))
        out = self.bn64(self.relu(self.conv3(out)))
        out = self.relu(self.conv4(out))
        out = self.bn128(self.relu(self.conv5(out)))
        out = self.relu(self.conv6(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.bn_out(self.fc2(h1))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 128, 4, 4)
        out = self.bn128(self.relu(self.deconv1(out)))
        out = self.relu(self.deconv2(out))
        out = self.bn64(self.relu(self.deconv3(out)))
        out = self.relu(self.deconv4(out))
        out = self.bn32(self.relu(self.deconv5(out)))
        out = self.sigmoid(self.deconv6(out))
        return out

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out