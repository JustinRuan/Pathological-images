from patches import DigitalSlide

slide = DigitalSlide()
tag = slide.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb")

if tag:
    ImageWidth, ImageHeight = slide.get_image_width_height_byScale(1)
    fullImage = slide.get_image_block(1, 0, 0, ImageWidth, ImageHeight)

tag = slide.release_slide_pointer()

slide.read_annotation('D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')
mask_img = slide.create_mask_image(1)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(4, 3))
ax = axes.ravel()

ax[0].imshow(fullImage)
ax[0].set_title("fullImage")
ax[1].imshow(mask_img)
ax[1].set_title("mask_img")
for a in ax.ravel():
    a.axis('off')

plt.show()
