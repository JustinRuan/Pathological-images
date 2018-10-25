#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-13'

"""
from core import Block
from core.Slice import Slice
import numpy as np
from skimage import draw
from skimage import color, morphology
from skimage.morphology import square

class ImageCone(object):

    def __init__(self, params):
        self._slice = Slice(params.KFB_SDK_PATH)
        self._params = params

    def open_slide(self, filename, ano_filename, id_string):
        '''
        打开切片文件
        :param filename: 切片文件名
        :param ano_filename: 切片所对应的标注文件名
        :param id_string: 切片编号
        :return: 是否成功打开切片
        '''
        path = "{}/{}".format(self._params.SLICES_ROOT_PATH, filename)
        self.slice_id = id_string
        tag = self._slice.open_slide(path, id_string)

        if tag:
            path = "{}/{}".format(self._params.SLICES_ROOT_PATH, ano_filename)
            self._slice.read_annotation(path)
            return True

        return False

    def get_fullimage_byScale(self, scale):
        '''
        得到指定倍镜下的全图
        :param scale: 倍镜数
        :return:
        '''
        w, h = self.get_image_width_height_byScale(scale)

        fullImage = self._slice.get_image_block(scale, w>>1, h>>1, w, h)
        return fullImage

    def get_image_width_height_byScale(self, scale):
        '''
        得到指定倍镜下，全图的大小
        :param scale: 倍镜数
        :return: 全图的宽，高
        '''
        if scale > 3:
            print("\a", "The size of image is too large")
            return

        w, h = self._slice.get_image_width_height_byScale(scale)
        return w, h

    def get_image_block(self, fScale, c_x, c_y, nWidth, nHeight):
        '''
        在指定坐标和倍镜下，提取切片的图块
        :param fScale: 倍镜数
        :param c_x: 中心x
        :param c_y: 中心y
        :param nWidth: 图块的宽
        :param nHeight: 图块的高
        :return: 返回一个图块对象
        '''
        data = self._slice.get_image_block_file(fScale, c_x, c_y, nWidth, nHeight)

        newBlock = Block.Block(self.slice_id, c_x, c_y, fScale, 0, nWidth, nHeight)
        newBlock.set_img_file(data)
        return newBlock
    
    def get_image_blocks_itor(self, fScale, set_x, set_y, nWidth, nHeight, batch_size):
        '''
        获得以种子点为左上角的图块的迭代器
        :param fScale: 倍镜数
        :param set_x: 中心点x的集合
        :param set_y: 中心点y的集合
        :param nWidth: 图块的宽
        :param nHeight: 图块的高
        :param batch_size: 每批的数量
        :return: 返回图块集合的迭代器
        '''
        n = 0
        images = []
        for x, y in zip(set_x, set_y):
            block = self.get_image_block(fScale, x, y, nWidth, nHeight)
            images.append(block.get_img())
            n = n + 1
            if n >= batch_size:
                yield images

                images = []
                n = 0

        if n > 0:
            return images

    def create_mask_image(self, scale, width):
        '''
        在设定的倍镜下，生成四种标注区的mask图像（NECL）
        :param scale: 指定的倍镜数
        :param width: 边缘区单边宽度
        :return: 对应的Mask图像
        '''
        w, h = self._slice.get_image_width_height_byScale(scale)
        '''
        癌变区代号 C， ano_TUMOR，将对应的标记区域，再腐蚀width宽。
        正常间质区代号 S， ano_STROMA，将对应的标记区域，再腐蚀width宽。
        淋巴区代号 L, ano_LYMPH, 将对应的标记区域,良性区域。
        边缘区代号 E， 在C和N，L之间的一定宽度的边缘，= ALL(有效区域) - C
       '''
        img = np.zeros((h, w), dtype=np.bool)
        for contour in self._slice.ano_TUMOR:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 1

        for contour in self._slice.ano_NORMAL:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 0

        C_img = morphology.binary_erosion(img, selem=square(width))
        SL_img = ~ morphology.binary_dilation(img, selem=square(width)) #淋巴区域包括在内
        E_img = np.bitwise_xor(np.ones((h, w), dtype=np.bool), np.bitwise_or(C_img, SL_img))

        img = np.zeros((h, w), dtype=np.bool)
        for contour in self._slice.ano_LYMPH:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 1
        L_img = morphology.binary_erosion(img, selem=square(8))
        S_img = np.bitwise_xor(SL_img, img)

        return C_img, S_img, E_img, L_img

    def get_effective_zone(self, scale):
        '''
        得到切片中，有效处理区域的Mask图像
        :param scale: 指定的倍镜
        :return: Mask图像
        '''
        fullImg = self.get_fullimage_byScale(scale)

        img = color.rgb2hsv(fullImg)
        # mask = np.ones(img.Shape, dtype=np.uint8)
        mask1 = (img[:, :, 2] < 0.9) & (img[:, :, 2] > 0.15)
        mask2 = (img[:, :, 1] < 0.9) & (img[:, :, 1] > 0.10)
        mask3 = (img[:, :, 0] < 0.9) & (img[:, :, 0] > 0.10)
        result = mask1 & mask2 & mask3

        result = morphology.binary_closing(result, square(10))
        result = morphology.binary_opening(result, square(20))
        result = morphology.binary_dilation(result, square(10))
        return result
