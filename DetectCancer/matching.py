#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, color, morphology, feature, measure, segmentation, transform


def matching():
    slide_ki67 = DigitalSlide()
    slide_ki67.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    slide_he = DigitalSlide()
    slide_he.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

    ImageWidth, ImageHeight = slide_he.get_image_width_height_byScale(2.5)
    he_img = slide_he.get_image_block(2.5, 0, 0, ImageWidth, ImageHeight)
    ImageWidth2, ImageHeight2 = slide_ki67.get_image_width_height_byScale(2.26)
    ki67_img = slide_ki67.get_image_block(2.26, 0, 0, ImageWidth2, ImageHeight2)

    slide_ki67.release_slide_pointer()
    slide_he.release_slide_pointer()
    return he_img, ki67_img


def matching2():
    slide_ki67 = DigitalSlide()
    slide_ki67.open_slide("D:/Study/breast/3Plus/17004930 KI-67_2017-07-29 09_48_32.kfb", "17004930")
    slide_he = DigitalSlide()
    slide_he.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

    # ImageWidth, ImageHeight = slide_he.get_image_width_height_byScale(2.5)
    he_img = slide_he.get_image_block(2.5 * 4, (1772 - 200) * 4, (2125 - 200) * 4, 400 * 4, 300 * 4)
    # ImageWidth2, ImageHeight2 = slide_ki67.get_image_width_height_byScale(2.25)
    ki67_img = slide_ki67.get_image_block(2.26 * 4, (963 - 200) * 4, (1589 - 200) * 4, 400 * 4, 300 * 4)

    slide_ki67.release_slide_pointer()
    slide_he.release_slide_pointer()
    return he_img, ki67_img


if __name__ == '__main__':
    he_img, ki67_img = matching2()
    # he_img.save('ke_img.jpg')
    # ki67_img.save('ki67_img.jpg')


    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    ax = axes.ravel()

    ax[0].imshow(he_img)
    ax[0].set_title("he_img")
    ax[1].imshow(ki67_img)
    ax[1].set_title("ki67_img")

    for a in ax.ravel():
        a.axis('off')

    plt.show()
