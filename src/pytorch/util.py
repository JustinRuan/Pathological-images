#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""
import torch
import torchvision
import numpy as np
from preparation.normalization import ImageNormalization

def get_image_blocks_itor(src_img, fScale, seeds, nWidth, nHeight, batch_size):
    '''
    获得以种子点为图块的迭代器
    :param fScale: 倍镜数
    :param seeds: 中心点集合
    :param nWidth: 图块的宽
    :param nHeight: 图块的高
    :param batch_size: 每批的数量
    :return: 返回图块集合的迭代器
    '''
    transform = torchvision.transforms.ToTensor()
    n = 0
    images = []
    for x, y in seeds:
        block = src_img.get_image_block(fScale, x, y, nWidth, nHeight)
        img = block.get_img() / 255
        img = transform(img).type(torch.FloatTensor)
        images.append(img)
        n = n + 1
        if n >= batch_size:
            img_tensor = torch.stack(images)
            yield img_tensor

            images = []
            n = 0

    if n > 0:
        img_tensor = torch.stack(images)
        yield img_tensor

def get_image_blocks_msc_itor(src_img, seeds_scale, seeds, nWidth, nHeight, batch_size):
    '''
    获得以种子点为图块的迭代器
    :param seeds_scale: 种子点的倍镜数
    :param seeds: 种子点
    :param nWidth: 图块的宽
    :param nHeight: 图块的高
    :param batch_size: 每批的数量
    :return: 返回图块集合的迭代器
    '''
    transform = torchvision.transforms.ToTensor()
    n = 0
    # fScale_list = [10, 20, 40]
    r10 = 10.0 / seeds_scale
    r20 = 2 * r10
    r40 = 4 * r10

    images = []
    for x, y in seeds:
        block10 = src_img.get_image_block(10, int(r10 * x), int(r10 * y), nWidth, nHeight)
        block20 = src_img.get_image_block(20, int(r20 * x), int(r20 * y), nWidth, nHeight)
        block40 = src_img.get_image_block(40, int(r40 * x), int(r40 * y), nWidth, nHeight)

        # img10 = ImageNormalization.normalize_mean( block10.get_img()) / 255
        # img20 = ImageNormalization.normalize_mean(block20.get_img()) / 255
        # img40 = ImageNormalization.normalize_mean(block40.get_img()) / 255

        img10 = block10.get_img() / 255
        img20 = block20.get_img() / 255
        img40 = block40.get_img() / 255
        img = np.concatenate((img10, img20, img40), axis=-1)

        img = transform(img).type(torch.FloatTensor)
        images.append(img)
        n = n + 1
        if n >= batch_size:
            img_tensor = torch.stack(images)
            yield img_tensor

            images = []
            n = 0

    if n > 0:
        img_tensor = torch.stack(images)
        yield img_tensor