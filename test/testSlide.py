from patches import DigitalSlide, get_roi, get_seeds, draw_seeds
import utils
import numpy as np


slide = DigitalSlide()
tag = slide.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

if tag:
    ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
    fullImage = slide.get_image_block(utils.GLOBAL_SCALE, 0, 0, ImageWidth, ImageHeight)

tag = slide.release_slide_pointer()

slide.read_annotation('D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')
mask_img = slide.create_mask_image(utils.GLOBAL_SCALE)

roi_img = get_roi(fullImage)
seeds = get_seeds(roi_img, 10)

patch_image = draw_seeds(fullImage, seeds, 32)




from matplotlib import pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(fullImage)
ax[0].set_title("fullImage")

ax[1].imshow(mask_img)
ax[1].set_title("mask_img")
ax[2].imshow(roi_img)
ax[2].set_title("roi_img")
ax[3].imshow(patch_image)
ax[3].set_title("patch_image")

for a in ax.ravel():
    a.axis('off')

plt.show()