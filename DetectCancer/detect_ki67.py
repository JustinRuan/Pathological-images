#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import caffe
from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, color, morphology, feature, measure, segmentation


def read_ki67(mask_img):
    slide = DigitalSlide()
    tag = slide.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")

    if tag:
        ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
        fullImage = slide.get_image_block(utils.EXTRACT_SCALE, 6400, 6400, 3840, 3200)
        # mask = mask_img[]
        A = color.rgb2hsv(fullImage)
        C = A[:, :, 1] > 0.7

        B = color.rgb2lab(fullImage)
        D = (B[:, :, 2] < -20)

        SE = morphology.square(5)
        C = morphology.binary_closing(C, SE)
        D = morphology.binary_closing(D, SE)

    tag = slide.release_slide_pointer()
    return fullImage, C, D


if __name__ == '__main__':
    mask = np.fromfile("he_result_img.bin", dtype=np.uint8)
    full_img, B, C = read_ki67(mask)
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    ax = axes.ravel()

    ax[0].imshow(full_img)
    ax[0].set_title("full_img")

    ax[1].imshow(full_img)
    ax[1].contour(B, [0.5], linewidths=0.5, colors='r')
    ax[1].contour(C, [0.5], linewidths=0.5, colors='b')
    ax[1].set_title("C")

    for a in ax.ravel():
        a.axis('off')

    plt.show()
