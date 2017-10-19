#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import caffe
from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds, Match_Slide
import utils
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, color, morphology, feature, measure, segmentation, transform
from scipy import signal


def read_ki67(roi_image):
    A = color.rgb2hsv(roi_image)
    C = A[:, :, 1] > 0.7

    B = color.rgb2lab(roi_image)
    D = (B[:, :, 2] < -18)

    SE = morphology.square(5)
    C = morphology.binary_closing(C, SE)
    D = morphology.binary_closing(D, SE)
    return C, D


def read_mask(mask, ratio, left_x, top_y, width, height):
    ImageHeight, ImageWidth = mask.shape
    large_mask = transform.resize(mask, (ImageHeight * ratio, ImageWidth * ratio))
    roi_mask = large_mask[top_y:top_y + height, left_x:left_x + width]
    roi_mask = roi_mask > 0.175
    return roi_mask


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
    ms = Match_Slide("/17004930 HE_2017-07-29 09_45_09.kfb", "/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    x = 1800 * 4
    y = 2000 * 4
    he_img, ki67_img = ms.get_Matched_Region(10, x, y, 800 * 2, 600 * 2)

    mask = np.fromfile("he_result_img (2292, 2681).bin", dtype=np.float)  # (1834, 2145) (2292, 2681)
    mask = np.reshape(mask, (2292, 2681))

    r = np.rint(10 / utils.GLOBAL_SCALE).astype(np.int)
    roi_mask = read_mask(mask, r, x, y, 800 * 2, 600 * 2)

    ki67, notki67 = read_ki67(ki67_img)
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
