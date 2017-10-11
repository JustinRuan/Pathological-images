#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import caffe
from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, color, morphology, feature, measure, segmentation, transform
from scipy import signal

def read_ki67(scale, left_x, top_y, width, height):
    slide = DigitalSlide()
    tag = slide.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")

    if tag:
        # ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
        # print(ImageWidth, ImageHeight)
        # space_patch = utils.PATCH_SIZE_LOW

        roi_image = slide.get_image_block(scale, left_x, top_y, width, height)
        A = color.rgb2hsv(roi_image)
        C = A[:, :, 1] > 0.7

        B = color.rgb2lab(roi_image)
        D = (B[:, :, 2] < -18)

        SE = morphology.square(5)
        C = morphology.binary_closing(C, SE)
        D = morphology.binary_closing(D, SE)

    tag = slide.release_slide_pointer()

    return roi_image, C, D


def read_mask(mask, left_x, top_y, width, height):
    slide = DigitalSlide()
    tag = slide.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")
    roi_image = slide.get_image_block(10, left_x, top_y, width, height)
    slide.release_slide_pointer()

    ImageHeight, ImageWidth = mask.shape
    large_mask = transform.resize(mask, (ImageHeight * 8, ImageWidth * 8))
    roi_mask = large_mask[top_y:top_y + height, left_x:left_x + width]
    roi_mask = roi_mask > 0.175
    return roi_image, roi_mask


def positive_hotmap(ki67, notki67, k):
    m = np.ones((k, k), dtype=np.float) / (k * k)

    mean_ki67 = signal.convolve2d(ki67, m)
    mean_notki67 = signal.convolve2d(notki67, m)

    # m = signal.gaussian(50, 10.0)
    result = (mean_ki67) / (mean_notki67 + 1e-3)

    SE = morphology.disk(3)
    result = morphology.dilation(result, SE)
    return result



if __name__ == '__main__':
    mask = np.fromfile("he_result_img (2292, 2681).bin", dtype=np.float)  # (1834, 2145) (2292, 2681)
    #
    mask = np.reshape(mask, (2292, 2681))

    he_img, roi_mask = read_mask(mask, (1772 - 200) * 4, (2125 - 200) * 4, 400 * 4, 300 * 4)

    ki67_img, ki67, notki67 = read_ki67(2.26 * 4, (963 - 200) * 4, (1589 - 200) * 4, 400 * 4, 300 * 4)
    ki67[~roi_mask] = 0

    result = positive_hotmap(ki67, notki67, 9)

    fig, axes = plt.subplots(2, 3, figsize=(4, 3))
    ax = axes.ravel()
    ax[0].imshow(ki67_img)
    ax[0].set_title("ki 67_img")

    ax[1].imshow(ki67_img)
    ax[1].contour(ki67, [0.5], linewidths=0.5, colors='r')
    ax[1].contour(notki67, [0.5], linewidths=0.5, colors='b')
    ax[1].set_title("Enhanced ki 67")

    ax[2].imshow(he_img)
    ax[2].set_title("he_img")

    ax[3].imshow(roi_mask)
    ax[3].set_title("roi_mask")

    ax[4].imshow(result, cmap=plt.cm.jet)
    ax[4].set_title("result hotmap of ki 67")
    for a in ax.ravel():
        a.axis('off')

    plt.show()
