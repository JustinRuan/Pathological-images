#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-17'

"""
from core import Slice, Block
import numpy as np
from skimage import draw
from skimage import color, morphology
from skimage.morphology import square

class ImageCone(object):

    def __init__(self, params):
        self._slice = Slice.Slice(params.KFB_SDK_PATH)
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

        fullImage = self._slice.get_image_block(scale, 0, 0, w, h)
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

    def get_image_block(self, fScale, sp_x, sp_y, nWidth, nHeight):
        '''
        在指定坐标和倍镜下，提取切片的图块
        :param fScale: 倍镜数
        :param sp_x: 左上角x
        :param sp_y: 左上角y
        :param nWidth: 图块的宽
        :param nHeight: 图块的高
        :return: 返回一个图块对象
        '''
        data = self._slice.get_image_block_file(fScale, sp_x, sp_y, nWidth, nHeight)

        newBlock = Block.Block(self.slice_id, sp_x, sp_y, fScale, 0, nWidth, nHeight)
        newBlock.set_img_file(data)
        return newBlock

    def create_mask_image(self, scale, mode):
        '''
        在设定的倍镜下，生成四种标注区的mask图像
        :param scale: 指定的倍镜数
        :param mode: 标注区类型
        :return: 对应的Mask图像
        '''
        w, h = self._slice.get_image_width_height_byScale(scale)
        img = np.zeros((h, w), dtype=np.bool)
        '''
        癌变精标区代号 TA， ano_TUMOR_A，在标记中直接给出。
        癌变粗标区代号 TR， ano_TUMOR_R，最终的区域从标记中计算得出， 
                            ano_TUMOR_R = 标记的TR - NR，即在CR中排除NR区域
        正常精标区代号 NA， ano_NORMAL_A，在标记中直接给出。
        正常粗标区代号 NR， ano_NORMAL_R，ano_NORMAL_R = ALL(有效区域) - 最终TR
       '''
        if mode == 'TA':
            for contour in self._slice.ano_TUMOR_A:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 1
        elif mode == 'TR':
            for contour in self._slice.ano_TUMOR_R:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 1

            for contour in self._slice.ano_NORMAL_R:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 0
        elif mode == 'NA':
            for contour in self._slice.ano_NORMAL_A:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 1
        else:  # mode == 'NR'
            img = np.ones((h, w), dtype=np.bool)

            for contour in self._slice.ano_TUMOR_R:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 0

            for contour in self._slice.ano_NORMAL_R:
                tumor_range = np.rint(contour * scale).astype(np.int)
                rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
                img[rr, cc] = 1

        return img

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

        result = morphology.binary_opening(result, square(20))
        result = morphology.binary_closing(result, square(5))
        return result
