#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-19'

"""
import torch
import torchvision
import numpy as np
from skimage.io import imread
from preparation.normalization import HistNormalization

def get_image_blocks_itor(src_img, fScale, seeds, nWidth, nHeight, batch_size, normalization = None):
    '''
    获得以种子点为图块的迭代器
    :param src_img: 切片图像
    :param fScale: 倍镜数
    :param seeds: 中心点集合
    :param nWidth: 图块的宽
    :param nHeight: 图块的高
    :param batch_size: 每批的数量
    :param normalization: 归一化算法 的类
    :return: 返回图块集合的迭代器
    '''
    transform = torchvision.transforms.ToTensor()
    n = 0
    images = []
    for x, y in seeds:
        block = src_img.get_image_block(fScale, x, y, nWidth, nHeight)

        img = block.get_img()
        if normalization is not None:
            img = normalization.normalize(img)
        else:
            img = img / 255.

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


# def get_image_blocks_batch_normalize_itor(src_img, fScale, seeds, nWidth, nHeight, batch_size, normalization):
#     '''
#     获得以种子点为图块的迭代器，使用全部种子点图块的归一化参数计算和归一化过程
#     :param src_img: 切片图像
#     :param fScale: 倍镜数
#     :param seeds: 中心点集合
#     :param nWidth: 图块的宽
#     :param nHeight: 图块的高
#     :param batch_size: 每批的数量
#     :param normalization: 归一化算法 的类
#     :return: 返回图块集合的迭代器
#     '''
#     assert normalization is not None, "Normalization is None!"
#
#     images = []
#     for x, y in seeds:
#         block = src_img.get_image_block(fScale, x, y, nWidth, nHeight)
#         img = block.get_img()
#         images.append(img)
#
#     normalization.prepare(images)
#     norm_images = normalization.normalize_on_batch(images)
#
#     transform = torchvision.transforms.ToTensor()
#     images = []
#     n = 0
#     for img in norm_images:
#         tmp_img = transform(img).type(torch.FloatTensor)
#         images.append(tmp_img)
#         n = n + 1
#         if n >= batch_size:
#             img_tensor = torch.stack(images)
#             yield img_tensor
#
#             images = []
#             n = 0
#
#     if n > 0:
#         img_tensor = torch.stack(images)
#         yield img_tensor

def get_image_blocks_batch_normalize_itor(src_img, fScale, seeds, nWidth, nHeight, batch_size, normalization, dynamic_update):
    '''
    获得以种子点为图块的迭代器，使用每批次图块的归一化参数计算和归一化过程
    :param src_img: 切片图像
    :param fScale: 倍镜数
    :param seeds: 中心点集合
    :param nWidth: 图块的宽
    :param nHeight: 图块的高
    :param batch_size: 每批的数量
    :param normalization: 归一化算法 的类
    :return: 返回图块集合的迭代器
    '''
    assert normalization is not None, "Normalization is None!"

    transform = torchvision.transforms.ToTensor()
    n = 0
    batch_images = []
    for x, y in seeds:
        block = src_img.get_image_block(fScale, x, y, nWidth, nHeight)
        img = block.get_img()
        batch_images.append(img)
        n = n + 1

        if n >= batch_size:
            if dynamic_update:
                normalization.prepare(batch_images)
            norm_images = normalization.normalize_on_batch(batch_images)
            temp = []
            for norm_img in norm_images:
                tmp_img = transform(norm_img).type(torch.FloatTensor)
                temp.append(tmp_img)

            img_tensor = torch.stack(temp)
            yield img_tensor

            batch_images = []
            n = 0

    if n > 0:
        if dynamic_update:
            normalization.prepare(batch_images)
        norm_images = normalization.normalize_on_batch(batch_images)
        temp = []
        for norm_img in norm_images:
            tmp_img = transform(norm_img).type(torch.FloatTensor)
            temp.append(tmp_img)

        img_tensor = torch.stack(temp)
        yield img_tensor

def get_image_file_batch_normalize_itor(image_filenames, y_set, batch_size, normalization, dynamic_update):
    '''
    获得以种子点为图块的迭代器，使用批量的归一化参数计算和归一化过程
    :param src_img: 切片图像
    :param fScale: 倍镜数
    :param seeds: 中心点集合
    :param nWidth: 图块的宽
    :param nHeight: 图块的高
    :param batch_size: 每批的数量
    :param normalization: 归一化算法 的类
    :param dynamic_update
    :return: 返回图块集合的迭代器
    '''
    assert normalization is not None, "Normalization is None!"
    transform = torchvision.transforms.ToTensor()
    n = 0
    batch_images = []
    batch_y = []
    for file_name, y_label in zip(image_filenames, y_set):
        img = imread(file_name)
        batch_images.append(img)
        batch_y.append(y_label)
        n = n + 1

        if n >= batch_size:
            if dynamic_update:
                normalization.prepare(batch_images)
            norm_images = normalization.normalize_on_batch(batch_images)
            temp = []
            for norm_img in norm_images:
                tmp_img = transform(norm_img).type(torch.FloatTensor)
                temp.append(tmp_img)

            img_tensor = torch.stack(temp)
            # y_tensor = torch.from_numpy(np.array(batch_y)).long()
            y_tensor = torch.tensor(batch_y, dtype = torch.long)
            yield img_tensor, y_tensor

            batch_images = []
            batch_y = []
            n = 0

    if n > 0:
        if dynamic_update:
            normalization.prepare(batch_images)
        norm_images = normalization.normalize_on_batch(batch_images)
        temp = []
        for norm_img in norm_images:
            tmp_img = transform(norm_img).type(torch.FloatTensor)
            temp.append(tmp_img)

        img_tensor = torch.stack(temp)
        y_tensor = torch.tensor(batch_y, dtype=torch.long)
        yield img_tensor, y_tensor

def get_image_blocks_dsc_itor(src_img, seeds_scale, seeds, nWidth, nHeight, batch_size):
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

    r20 = 20 / seeds_scale
    r40 = 40 / seeds_scale

    images_x20 = []
    images_x40 = []
    for x, y in seeds:
        block20 = src_img.get_image_block(20, int(r20 * x), int(r20 * y), nWidth, nHeight)
        block40 = src_img.get_image_block(40, int(r40 * x), int(r40 * y), nWidth, nHeight)

        img20 = block20.get_img() / 255
        img40 = block40.get_img() / 255

        images_x20.append(transform(img20).type(torch.FloatTensor))
        images_x40.append(transform(img40).type(torch.FloatTensor))

        n = n + 1
        if n >= batch_size:
            img_x20_tensor = torch.stack(images_x20)
            img_x40_tensor = torch.stack(images_x40)
            yield img_x20_tensor, img_x40_tensor

            images_x20 = []
            images_x40 = []
            n = 0

    if n > 0:
        img_x20_tensor = torch.stack(images_x20)
        img_x40_tensor = torch.stack(images_x40)
        yield img_x20_tensor, img_x40_tensor


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