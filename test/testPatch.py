#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np

slide = DigitalSlide()
tag = slide.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

if tag:
    ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
    fullImage = slide.get_image_block(utils.GLOBAL_SCALE, 0, 0, ImageWidth, ImageHeight)

slide.read_annotation('D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')

mask_img = slide.create_mask_image(utils.GLOBAL_SCALE)

ex_patch = Patch()
ex_patch.get_roi_seeds(fullImage, 5)

ex_patch.extract_patches(slide, mask_img, utils.EXTRACT_SCALE, utils.PATCH_SIZE_HIGH)

tag = slide.release_slide_pointer()
