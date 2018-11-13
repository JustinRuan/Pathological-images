#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-11'

"""

# import sys
# sys.path.append("C:/Program Files/ASAP 1.8/bin")



# mr_image = reader.open('camelyon17/centre_0/patient_000_node_0.tif')
# level = 2
# ds = mr_image.getLevelDownsample(level)
# image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), 300, 200, level)


# import multiresolutionimageinterface as mir
# reader = mir.MultiResolutionImageReader()
# mr_image = reader.open('camelyon17/centre_0/patient_010_node_4.tif')
# annotation_list = mir.AnnotationList()
# xml_repository = mir.XmlRepository(annotation_list)
# xml_repository.setSource('camelyon17/centre_0/patient_010_node_4.xml')
# xml_repository.load()
# annotation_mask = mir.AnnotationToMask()
# camelyon17_type_mask = True
# label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
# conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
# annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)


import numpy as np
import multiresolutionimageinterface as mir

class ASAP_Slide(object):
    def __init__(self):
        self.reader = mir.MultiResolutionImageReader()
        self.Normalization_coefficient = 20
        return

    def open_slide(self, filename, id_string):
        self.img = self.reader.open(filename)
        self._id = id_string

        if self.img is None:
            return False
        else:
            return True

    def get_id(self):
        '''
        得取切片的编号
        :return: 返回切片编号
        '''
        return self._id

    def get_image_width_height_byScale(self, scale):
        '''
        得到指定倍镜下，整个切片图像的大小
        :param scale: 提取图像所在的倍镜数, 归一化到倍镜数
        :return: 图像的大小，宽（x）高（y）
        '''
        # ds = self.img.getLevelDownsample(scale)
        w, h = self.img.getDimensions()
        ImageWidth = np.rint(w // self.Normalization_coefficient * scale).astype(np.int)
        ImageHeight = np.rint(h // self.Normalization_coefficient * scale).astype(np.int)
        return ImageWidth, ImageHeight

    def get_image_block(self, fScale, c_x, c_y, nWidth, nHeight):

        # 从中心坐标移动到左上角坐标
        # sp_x = c_x - (nWidth >> 1)
        # sp_y = c_y - (nHeight >> 1)
        sp_x = c_x
        sp_y = c_y

        level = int(self.Normalization_coefficient / fScale)

        ds = self.img.getLevelDownsample(level)
        # image_patch = self.img.getUCharPatch(int(sp_x), int(sp_y), int(nWidth), int(nHeight), level)
        image_patch = self.img.getUCharPatch(int(sp_x), int(sp_y), int(nWidth), int(nHeight), level)
        return image_patch







