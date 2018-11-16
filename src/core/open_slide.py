#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-13'

"""
# import os
# os.environ['PATH'] = "C:/Work/openslide/bin" + ";" + os.environ['PATH'] # 一定要路径加到前面去

import numpy as np
import math
import openslide
from openslide import open_slide, ImageSlide
import io
import xml.dom.minidom
from skimage import draw
from skimage import color, morphology
from skimage.morphology import square

class Open_Slide(object):
    def __init__(self):
        self.Normalization_coefficient = 40 # level 1, 50um, 20倍镜, level 0. 25um, 40倍
        return

    def open_slide(self, filename, id_string):
        self.img = open_slide(filename)
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
        w, h = self.img.dimensions # A (width, height) tuple for level 0 of the slide.
        ImageWidth = np.rint(w / self.Normalization_coefficient * scale).astype(np.int)
        ImageHeight = np.rint(h / self.Normalization_coefficient * scale).astype(np.int)
        return ImageWidth, ImageHeight

    def convert_scale_to_level(self, scale):
        return int(math.log(self.Normalization_coefficient / scale, 2))

    def get_image_block(self, c_scale, c_x, c_y, nWidth, nHeight):
        '''
        提取图块
        :param c_scale: 提取用的倍镜数
        :param c_x: 该倍镜下的x坐标
        :param c_y: 该倍镜下的y坐标
        :param nWidth: 该倍镜下的宽度
        :param nHeight: 该倍镜下的高度
        :return: 图块
        '''
        # 从中心坐标移动到左上角坐标, sp_x,sp_y是40倍镜下的坐标
        amp = self.Normalization_coefficient / c_scale
        sp_x = int((c_x - (nWidth >> 1)) * amp)
        sp_y = int((c_y - (nHeight >> 1)) * amp)
        # sp_x = c_x
        # sp_y = c_y

        level = self.convert_scale_to_level(c_scale)
        image_patch = self.img.read_region((sp_x, sp_y), level, (nWidth, nHeight))
        return np.array(image_patch)

    def get_thumbnail(self, scale):
        width, height = self.get_image_width_height_byScale(scale)
        return self.img.get_thumbnail((width, height))


    '''
    CAMELYON16 
    Annotations belonging to group "0" and "1" represent tumor areas and 
    annotations within group "2" are non-tumor areas which have been cut-out from 
    the original annotations in the first two group.
    '''
    def read_annotation(self, filename):
        '''
        读取标注文件
        :param filename: 切片标注文件名
        :return:
        '''
        self.ano = {"TUMOR": [], "NORMAL": []}

        # 使用minidom解析器打开 XML 文档
        fp = open(filename, 'r', encoding="utf-8")
        content = fp.read()
        fp.close()

        # content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')

        DOMTree = xml.dom.minidom.parseString(content)
        collection = DOMTree.documentElement

        Regions = collection.getElementsByTagName("Annotation")

        for Region in Regions:
            if Region.hasAttribute("PartOfGroup"):
                Vertices = Region.getElementsByTagName("Coordinate")
                range_type = Region.getAttribute("PartOfGroup").strip()

                posArray = np.zeros((len(Vertices), 2))
                i = 0
                for item in Vertices:
                    # 归一化到1倍镜下坐标
                    posArray[i][0] = float(item.getAttribute("X")) / self.Normalization_coefficient
                    posArray[i][1] = float(item.getAttribute("Y")) / self.Normalization_coefficient
                    i += 1

                if range_type == "_0" or range_type == "_1" :
                    self.ano["TUMOR"].append(posArray)
                elif range_type == "_2":
                    self.ano["NORMAL"].append(posArray)

        return

    def create_mask_image(self, scale, edge_width):
        '''
        在设定的倍镜下，生成四种标注区的mask图像（NECL）
        :param scale: 指定的倍镜数
        :param edge_width: 边缘区单边宽度
        :return: 对应的Mask图像
        '''
        w, h = self.get_image_width_height_byScale(scale)
        '''
        癌变区代号 C， ano_TUMOR，将对应的标记区域，再腐蚀width宽。
        正常间质区代号 S， ano_STROMA，将对应的标记区域，再腐蚀width宽。
        边缘区代号 E， 在C和N，L之间的一定宽度的边缘，= ALL(有效区域) - C
       '''
        img = np.zeros((h, w), dtype=np.bool)

        for contour in self.ano["TUMOR"]:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 1

        for contour in self.ano["NORMAL"]:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 0

        if edge_width > 1:
            C_img = morphology.binary_erosion(img, selem=square(edge_width))
            N_img = ~ morphology.binary_dilation(img, selem=square(edge_width))
            E_img = np.bitwise_xor(np.ones((h, w), dtype=np.bool), np.bitwise_or(C_img, N_img))
        else:
            C_img = img
            N_img = ~img
            E_img = np.zeros((h, w), dtype=np.bool)

        return {"C": C_img, "N": N_img, "E": E_img}




