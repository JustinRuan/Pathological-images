#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""
import torch
import torchvision

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