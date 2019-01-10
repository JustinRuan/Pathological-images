#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-17'

"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable

# class Autoencoder(nn.Module):
#     def __init__(self, out_dim):
#         super(Autoencoder, self).__init__()
#         intermediate_size = 128
#
#         # Encoder
#         # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) #  32 x 32
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0) # 16 x 16
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 16 x 16
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0) # 8 x 8
#         self.conv5= nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 8 x 8
#         self.conv6 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0) # 4 x 4
#         self.fc1 = nn.Linear(4 * 4 * 128, intermediate_size)
#         self.fc2 = nn.Linear(intermediate_size, out_dim)
#
#         # Decoder
#         self.fc3 = nn.Linear(out_dim, intermediate_size)
#         # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
#         self.fc4 = nn.Linear(intermediate_size, 2048) # 4x4x128
#         self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0) # 8 x 8
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1) # 8 x 8
#         self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0) # 16 x 16
#         self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1) # 16 x 16
#         self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0) # 32 x 32
#         self.deconv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1) # 32x32
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.bn32 = nn.BatchNorm2d(32, affine=False)
#         self.bn64 = nn.BatchNorm2d(64, affine=False)
#         self.bn128 = nn.BatchNorm2d(128, affine=False)
#         self.bn_out = nn.BatchNorm1d(out_dim, affine=False)
#
#     def encode(self, x):
#         out = self.bn32(self.relu(self.conv1(x)))
#         out = self.relu(self.conv2(out))
#         out = self.bn64(self.relu(self.conv3(out)))
#         out = self.relu(self.conv4(out))
#         out = self.bn128(self.relu(self.conv5(out)))
#         out = self.relu(self.conv6(out))
#         out = out.view(out.size(0), -1)
#         h1 = self.relu(self.fc1(out))
#         return self.bn_out(self.fc2(h1))
#
#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         out = self.relu(self.fc4(h3))
#         out = out.view(out.size(0), 128, 4, 4)
#         out = self.bn128(self.relu(self.deconv1(out)))
#         out = self.relu(self.deconv2(out))
#         out = self.bn64(self.relu(self.deconv3(out)))
#         out = self.relu(self.deconv4(out))
#         out = self.bn32(self.relu(self.deconv5(out)))
#         out = self.sigmoid(self.deconv6(out))
#         return out
#
#     def forward(self, x):
#         out = self.encode(x)
#         out = self.decode(out)
#         return out

# 与上面的类相比，就是改变了写法，模块化
# convolutional Auto-Encoder, output_z = False
# contractive autoencoder, CAE, output_z = True
class Autoencoder(nn.Module):
    def __init__(self, out_dim, output_z = False):
        super(Autoencoder, self).__init__()
        intermediate_size = 128
        self.output_z = output_z

        # Encoder
        # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
        self.encoder = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)), # 32 x 32 => 32 x 32
                ("relu1", nn.ReLU()),
                ("bn_c1", nn.BatchNorm2d(32, affine=False)),
                ("conv2", nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)), # 16 x 16
                ("relu2", nn.ReLU()),
                ("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)), # 16 x 16
                ("relu3", nn.ReLU()),
                ("bn_c3", nn.BatchNorm2d(64, affine=False)),
                ("conv4", nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0)), # 8 x 8
                ("relu4", nn.ReLU()),
                ("conv5", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)), # 8 x 8
                ("relu5", nn.ReLU()),
                ("bn_c5", nn.BatchNorm2d(128, affine=False)),
                ("conv6", nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)), # 4 x 4
                ("relu6", nn.ReLU()),
            ])
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
        )

        # Decoder
        # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
        self.decoder_fc = nn.Sequential(
            nn.Linear(out_dim, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, 2048),
            nn.ReLU(),
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

    def encode(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        return self.encoder_fc(out)

    def decode(self, z):
        out = self.decoder_fc(z)
        out = out.view(out.size(0), 128, 4, 4)
        out = self.decoder(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        if self.output_z:
            return out, z
        else:
            return out

# Variational Auto-Encoder,VAE
class VAE(nn.Module):
    def __init__(self, out_dim):
        super(VAE, self).__init__()
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

        # Latent space
        self.fc21 = nn.Linear(intermediate_size, out_dim)
        self.fc22 = nn.Linear(intermediate_size, out_dim)

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

        # return self.bn_out(self.fc21(h1)), self.bn_out(self.fc22(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#
#         # Encoder
#         self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(16 * 16 * 32, args.intermediate_size)
#
#         # Latent space
#         self.fc21 = nn.Linear(args.intermediate_size, args.hidden_size)
#         self.fc22 = nn.Linear(args.intermediate_size, args.hidden_size)
#
#         # Decoder
#         self.fc3 = nn.Linear(args.hidden_size, args.intermediate_size)
#         self.fc4 = nn.Linear(args.intermediate_size, 8192)
#         self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
#         self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def encode(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.relu(self.conv3(out))
#         out = self.relu(self.conv4(out))
#         out = out.view(out.size(0), -1)
#         h1 = self.relu(self.fc1(out))
#         return self.fc21(h1), self.fc22(h1)
#
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = Variable(std.data.new(std.size()).normal_())
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         out = self.relu(self.fc4(h3))
#         # import pdb; pdb.set_trace()
#         out = out.view(out.size(0), 32, 16, 16)
#         out = self.relu(self.deconv1(out))
#         out = self.relu(self.deconv2(out))
#         out = self.relu(self.deconv3(out))
#         out = self.sigmoid(self.conv5(out))
#         return out
#
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # contractive autoencoder, CAE
# class CAE(nn.Module):
#     def __init__(self, out_dim):
#         super(CAE, self).__init__()
#         intermediate_size = 128
#
#         # Encoder
#         # output_size = 1 + (input_size + 2*padding - kernel_size)/stride
#         self.encoder = nn.Sequential(
#             OrderedDict([
#                 ("conv1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)), # 32 x 32 => 32 x 32
#                 ("relu1", nn.ReLU()),
#                 ("bn_c1", nn.BatchNorm2d(32, affine=False)),
#                 ("conv2", nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)), # 16 x 16
#                 ("relu2", nn.ReLU()),
#                 ("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)), # 16 x 16
#                 ("relu3", nn.ReLU()),
#                 ("bn_c3", nn.BatchNorm2d(64, affine=False)),
#                 ("conv4", nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0)), # 8 x 8
#                 ("relu4", nn.ReLU()),
#                 ("conv5", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)), # 8 x 8
#                 ("relu5", nn.ReLU()),
#                 ("bn_c5", nn.BatchNorm2d(128, affine=False)),
#                 ("conv6", nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)), # 4 x 4
#                 ("relu6", nn.ReLU()),
#             ])
#         )
#         self.encoder_fc = nn.Sequential(
#             OrderedDict([
#                 ("fc1", nn.Linear(4 * 4 * 128, intermediate_size)),
#                 ("relu1", nn.ReLU()),
#                 ("fc2",nn.Linear(intermediate_size, out_dim)),
#                 ("bn_fc", nn.BatchNorm1d(out_dim, affine=False)),
#             ])
#         )
#
#         # Decoder
#         # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(out_dim, intermediate_size),
#             nn.ReLU(),
#             nn.Linear(intermediate_size, 2048),
#             nn.ReLU(),
#         )
#
#         self.decoder =  nn.Sequential(
#             OrderedDict([
#                 ("deconv1", nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)), #  4x4 => 8 x 8
#                 ("relu1",   nn.ReLU()),
#                 ("bn_c1",   nn.BatchNorm2d(128, affine=False)),
#                 ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)), # 8 x 8
#                 ("relu2",   nn.ReLU()),
#                 ("deconv3", nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)), # 16 x 16
#                 ("relu3",   nn.ReLU()),
#                 ("bn_c3",   nn.BatchNorm2d(64, affine=False)),
#                 ("deconv4", nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)), # 16 x 16
#                 ("relu4",   nn.ReLU()),
#                 ("deconv5", nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)), # 32 x 32
#                 ("relu5",   nn.ReLU()),
#                 ("bn_c5",   nn.BatchNorm2d(32, affine=False)),
#                 ("deconv6", nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)), # 32 x3 2
#                 ("sigmoid", nn.Sigmoid()),
#             ])
#         )
#
#     def encode(self, x):
#         out = self.encoder(x)
#         out = out.view(out.size(0), -1)
#         return self.encoder_fc(out)
#
#     def decode(self, z):
#         out = self.decoder_fc(z)
#         out = out.view(out.size(0), 128, 4, 4)
#         out = self.decoder(out)
#         return out
#
#     def forward(self, x):
#         h1 = self.encode(x)
#         h2 = self.decode(h1)
#         return h1, h2



