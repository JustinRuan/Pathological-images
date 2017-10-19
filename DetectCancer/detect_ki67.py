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
from scipy.ndimage.filters import gaussian_filter

def read_ki67(roi_image):
    A = color.rgb2hsv(roi_image)
    C = A[:, :, 1] > 0.7

    B = color.rgb2lab(roi_image)
    D = (B[:, :, 2] < -18)

    SE = morphology.square(5)
    C = morphology.binary_closing(C, SE)
    D = morphology.binary_closing(D, SE)
    return C, D


def read_mask(mask, ratio):
    ImageHeight, ImageWidth = mask.shape
    large_mask = transform.resize(mask, (ImageHeight * ratio, ImageWidth * ratio))
    return large_mask


def get_roi_mask(large_mask, left_x, top_y, width, height):
    roi_mask = large_mask[top_y:top_y + height, left_x:left_x + width]
    roi_mask = roi_mask > 0.018
    return roi_mask

def positive_hotmap(ki67, notki67, k):
    m = np.ones((k, k), dtype=np.float) / (k * k)

    mean_ki67 = signal.convolve2d(ki67, m)
    mean_notki67 = signal.convolve2d(notki67, m)

    ratio_array = (mean_ki67) / (mean_notki67 + 1e-3)
    # max_value = np.max(ratio_array)
    # if max_value > 0:
    #     ratio_array = ratio_array / max_value

    SE = morphology.disk(10)
    morp_array = morphology.dilation(ratio_array, SE)

    result = gaussian_filter(morp_array, sigma=5, truncate=4.0)
    return result


def detect_ki67(ms):
    mask = np.fromfile("he_result_img (2292, 2681).bin", dtype=np.float)  # (1834, 2145) (2292, 2681)
    mask = np.reshape(mask, (2292, 2681))

    r = np.rint(10 / utils.GLOBAL_SCALE).astype(np.int)
    large_mask = read_mask(mask, r)

    x0, y0, x1, y1 = ms.get_overlap_region(1)
    he_global_img, ki67_global_img = ms.get_Matched_Region(1, x0, y0, x1 - x0, y1 - y0)

    x0, y0, x1, y1 = ms.get_overlap_region(10)

    small_width = np.rint((x1 - x0) / 16).astype(np.int32)
    small_height = np.rint((y1 - y0) / 16).astype(np.int32)
    width = 4 * small_width  # (np.rint((x1 - x0) / 16) * 4).astype(np.int32)
    height = 4 * small_height  # (np.rint((y1 - y0) / 16) * 4).astype(np.int32)

    global_result = np.zeros((height, width), dtype=np.float)

    for n in range(4):
        for m in range(4):
            xx = x0 + m * width
            yy = y0 + n * height

            roi_mask = get_roi_mask(large_mask, xx, yy, width, height)
            he_img, ki67_img = ms.get_Matched_Region(10, xx, yy, width, height)

            ki67, notki67 = read_ki67(ki67_img)
            ki67[~roi_mask] = 0

            result = positive_hotmap(ki67, notki67, 9)
            small_reault = transform.resize(result, (small_height, small_width))
            global_result[n * small_height:(n + 1) * small_height, m * small_width:(m + 1) * small_width] = small_reault
            print((n, m))

    return he_global_img, ki67_global_img, global_result


def test1(ms):
    x = 2500 * 4
    y = 1250 * 4
    width = 2000
    height = 2000
    he_img, ki67_img = ms.get_Matched_Region(10, x, y, width, height)

    mask = np.fromfile("he_result_img (2292, 2681).bin", dtype=np.float)  # (1834, 2145) (2292, 2681)
    mask = np.reshape(mask, (2292, 2681))

    r = np.rint(10 / utils.GLOBAL_SCALE).astype(np.int)
    large_mask = read_mask(mask, r)
    roi_mask = get_roi_mask(large_mask, x, y, width, height)

    ki67, notki67 = read_ki67(ki67_img)
    ki67[~roi_mask] = 0

    result = positive_hotmap(ki67, notki67, 9)
    result = transform.resize(result, (500, 500))

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


if __name__ == '__main__':
    ms = Match_Slide("/17004930 HE_2017-07-29 09_45_09.kfb", "/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    # test1(ms)

    he_global_img, ki67_global_img, global_result = detect_ki67(ms)

    fig, axes = plt.subplots(1, 3, figsize=(4, 3))
    ax = axes.ravel()
    ax[0].imshow(he_global_img)
    ax[0].set_title("he_global_img")

    ax[1].imshow(ki67_global_img)
    ax[1].set_title("ki67_global_img")

    ax[2].imshow(global_result)
    ax[2].set_title("global_result")

    for a in ax.ravel():
        a.axis('off')

    plt.show()
