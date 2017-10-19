#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, color, morphology, feature, measure, segmentation, transform, exposure
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches, BRIEF)


class Match_Slide(object):
    def __init__(self, he_filename, ki67_filename, file_id):
        self.slide_ki67 = DigitalSlide()
        self.slide_ki67.open_slide(utils.SLIDES_PATH + ki67_filename, file_id)
        x0, y0 = self.slide_ki67.read_remark(utils.SLIDES_PATH + ki67_filename + '.Ano')

        self.slide_he = DigitalSlide()
        self.slide_he.open_slide(utils.SLIDES_PATH + he_filename, file_id)
        x1, y1 = self.slide_he.read_remark(utils.SLIDES_PATH + he_filename + '.Ano')

        # 1倍镜下的 float偏移量
        self.control_pointX = x0 - x1
        self.control_pointY = y0 - y1

    def get_Matched_Region(self, scale, he_x, he_y, width, height):
        xx = np.rint(scale * self.control_pointX).astype(np.int32)
        yy = np.rint(scale * self.control_pointY).astype(np.int32)

        ki67_x = he_x + xx
        ki67_y = he_y + yy

        he_img = self.slide_he.get_image_block(scale, he_x, he_y, width, height)
        ki67_img = self.slide_ki67.get_image_block(scale, ki67_x, ki67_y, width, height)

        return he_img, ki67_img

    def __del__(self):
        self.slide_ki67.release_slide_pointer()
        self.slide_he.release_slide_pointer()

    def get_overlap_region(self, scale):
        he_width, he_height = self.slide_he.get_image_width_height_byScale(scale)
        ki67_width, ki67_height = self.slide_ki67.get_image_width_height_byScale(scale)
        he_x0 = 0
        he_y0 = 0
        he_x1 = he_width
        he_y1 = he_height

        ki67_x0 = -scale * self.control_pointX
        ki67_y0 = -scale * self.control_pointY
        ki67_x1 = -scale * self.control_pointX + ki67_width
        ki67_y1 = -scale * self.control_pointY + ki67_height

        # 以he的图像为参考坐标系
        new_x0 = np.rint(max(he_x0, ki67_x0)).astype(np.int32)
        new_y0 = np.rint(max(he_y0, ki67_y0)).astype(np.int32)
        new_x1 = np.rint(min(he_x1, ki67_x1)).astype(np.int32)
        new_y1 = np.rint(min(he_y1, ki67_y1)).astype(np.int32)
        return new_x0, new_y0, new_x1, new_y1

if __name__ == '__main__':
    ms = Match_Slide("/17004930 HE_2017-07-29 09_45_09.kfb", "/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    x0, y0, x1, y1 = ms.get_overlap_region(1)
    print((x0, y0, x1, y1))

    # he_img, ki67_img = ms.get_Matched_Region(10, 1500 * 4, 2000 * 4, 800 * 2, 600 * 2)
    he_img, ki67_img = ms.get_Matched_Region(1, x0, y0, x1 - x0, y1 - y0)

    print(ms.get_overlap_region(1))

    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    ax = axes.ravel()

    ax[0].imshow(he_img)
    ax[0].set_title("he_img")
    ax[1].imshow(ki67_img)
    ax[1].set_title("ki67_img")

    for a in ax.ravel():
        a.axis('off')

    plt.show()
