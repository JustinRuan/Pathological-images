#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-13'

"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
#                  input_shape=input_shape, kernel_initializer='random_uniform'))  # (128, 128, 3)
# model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#
# model.add(Conv2D(48, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#
# model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
#
# model.add(GlobalAveragePooling2D())
#
# # 全连接层
# model.add(Dense(top_units, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

class Simple_CNN(nn.Module):

    def __init__(self, num_classes, image_size):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(32),  # activation
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(32),
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, image_size / 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(48),
            nn.MaxPool2d(kernel_size=2)  # image_size / 4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.ReLU(64),
            nn.MaxPool2d(kernel_size=2) # image_size / 8
        )
        self.gap = nn.AvgPool2d(kernel_size=image_size >> 3)
        self.dense = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(256),
        )
        self.out = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图成 (batch_size, .....)
        x = self.dense(x)

        output = self.out(x)
        return output


