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


def matching():
    slide_ki67 = DigitalSlide()
    slide_ki67.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    slide_he = DigitalSlide()
    slide_he.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

    ImageWidth, ImageHeight = slide_he.get_image_width_height_byScale(2.5)
    he_img = slide_he.get_image_block(2.5, 0, 0, ImageWidth, ImageHeight)
    ImageWidth2, ImageHeight2 = slide_ki67.get_image_width_height_byScale(2.5)
    ki67_img = slide_ki67.get_image_block(2.5, 0, 0, ImageWidth2, ImageHeight2)

    slide_ki67.release_slide_pointer()
    slide_he.release_slide_pointer()
    return he_img, ki67_img


def matching2():
    slide_ki67 = DigitalSlide()
    slide_ki67.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    x0, y0 = slide_ki67.read_remark('D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb.Ano')

    slide_he = DigitalSlide()
    slide_he.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")
    x1, y1 = slide_he.read_remark('D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')

    xx = np.rint(10 * (x0 - x1)).astype(np.int32)
    yy = np.rint(10 * (y0 - y1)).astype(np.int32)

    # # ImageWidth, ImageHeight = slide_he.get_image_width_height_byScale(2.5)
    # he_img = slide_he.get_image_block(2.5 * 4, (4002 - 000) * 4, (1220 - 000) * 4, 800 * 2, 600 * 2)
    # # ImageWidth2, ImageHeight2 = slide_ki67.get_image_width_height_byScale(2.25)
    # ki67_img = slide_ki67.get_image_block(2.5 * 4, (3268 - 000) * 4, (826 - 000) * 4, 800 * 2, 600 * 2)
    print(xx, yy)

    he_img = slide_he.get_image_block(10, 4000 * 4, 2220 * 4, 800 * 2, 600 * 2)
    ki67_img = slide_ki67.get_image_block(10, (4000 * 4 + xx), (2220 * 4 + yy), 800 * 2, 600 * 2)

    slide_ki67.release_slide_pointer()
    slide_he.release_slide_pointer()
    return he_img, ki67_img


def matching3():
    slide_ki67 = DigitalSlide()
    slide_ki67.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    slide_he = DigitalSlide()
    slide_he.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

    ImageWidth, ImageHeight = slide_he.get_image_width_height_byScale(1)
    he_img = slide_he.get_image_block(1, 0, 0, ImageWidth, ImageHeight)
    ImageWidth2, ImageHeight2 = slide_ki67.get_image_width_height_byScale(1)
    ki67_img = slide_ki67.get_image_block(1, 0, 0, ImageWidth2, ImageHeight2)

    slide_ki67.release_slide_pointer()
    slide_he.release_slide_pointer()
    return he_img, ki67_img


def find_keypoints(he_img, ki67_img):
    img1 = np.array(he_img)
    img2 = np.array(ki67_img)

    img1 = color.rgb2gray(img1)
    img2 = color.rgb2gray(img2)

    img1 = exposure.adjust_gamma(img1, 0.9)
    img2 = exposure.adjust_gamma(img2, 1.5)
    # img1 = exposure.equalize_hist(img1)
    # img2 = exposure.equalize_hist(img2)

    descriptor_extractor = ORB(downscale=2.0, n_scales=4,
                               n_keypoints=50, fast_n=9, fast_threshold=0.08,
                               harris_k=0.04)

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # keypoints1 = corner_peaks(corner_harris(img1, k=0.01), min_distance=20)
    # keypoints2 = corner_peaks(corner_harris(img2, k=0.01), min_distance=20)
    #
    # extractor = BRIEF()
    #
    # extractor.extract(img1, keypoints1)
    # keypoints1 = keypoints1[extractor.mask]
    # descriptors1 = extractor.descriptors
    #
    # extractor.extract(img2, keypoints2)
    # keypoints2 = keypoints2[extractor.mask]
    # descriptors2 = extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    fig = plt.gcf()
    plot_matches(fig.gca(), img1, img2, keypoints1, keypoints2, matches12, matches_color='r')
    # ax[0].axis('off')
    return


if __name__ == '__main__':
    he_img, ki67_img = matching2()
    # he_img.save('ke_img.jpg')
    # ki67_img.save('ki67_img.jpg')

    # find_keypoints(he_img, ki67_img)

    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    ax = axes.ravel()

    ax[0].imshow(he_img)
    ax[0].set_title("he_img")
    ax[1].imshow(ki67_img)
    ax[1].set_title("ki67_img")

    for a in ax.ravel():
        a.axis('off')

    plt.show()
