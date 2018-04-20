from Function import DigitalSlide, Patch

import utils

# 读取数字切片图像及其标注信息
# 依照标注信息进行样本切片


slide = DigitalSlide()
# 读取数字全扫描切片图像
tag = slide.open_slide("H:/TumorSlide/3plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

if tag:
    ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
    fullImage = slide.get_image_block(utils.GLOBAL_SCALE, 0, 0, ImageWidth, ImageHeight)

# 读取标注信息
slide.read_annotation('H:/TumorSlide/3plus/17004930 HE_2017-07-29 09_45_09.kfb.Ano')

mask_img = slide.create_mask_image(utils.GLOBAL_SCALE)

ex_patch = Patch()
ex_patch.get_roi_seeds(fullImage, utils.EXTRACT_PATCH_DIST)

ex_patch.extract_patches(slide, mask_img, utils.EXTRACT_SCALE, utils.PATCH_SIZE_HIGH)

tag = slide.release_slide_pointer()
