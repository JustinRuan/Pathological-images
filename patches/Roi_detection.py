import numpy as np
from skimage import color, morphology
from skimage.morphology import square
from PIL import ImageDraw
import utils

def get_roi(src_img):
    img = color.rgb2hsv(src_img)
    # mask = np.ones(img.Shape, dtype=np.uint8)
    mask1 = (img[:, :, 2] < 0.9) & (img[:, :, 2] > 0.15)
    mask2 = (img[:, :, 1] < 0.9) & (img[:, :, 1] > 0.10)
    mask3 = (img[:, :, 0] < 0.9) & (img[:, :, 0] > 0.10)
    result = mask1 & mask2 & mask3

    result = morphology.binary_opening(result, square(20))
    result = morphology.binary_closing(result, square(5))

    return result


def get_seeds(src_img, distance):
    patch_size = utils.PATCH_SIZE_LOW
    seed_img = morphology.binary_erosion(src_img, square(patch_size))#灰度图像腐蚀，图像中物体会收缩/细化：https://wenku.baidu.com/view/c600c8d1360cba1aa811da73.html
    seed_img = morphology.binary_erosion(seed_img, square(8))  # 留边

    space_patch = distance
    pos = seed_img.nonzero()
    y = (np.rint(pos[0] / space_patch + 0.5) * space_patch).astype(np.int32)  # row
    x = (np.rint(pos[1] / space_patch + 0.5) * space_patch).astype(np.int32)  # col

    result = set()
    for xx, yy in zip(x, y):
        result.add((xx, yy))

    return result


def draw_seeds(src_img, seeds, patch_size):
    draw = ImageDraw.Draw(src_img)
    half = patch_size / 2
    for (x, y) in seeds:
        draw.rectangle([x - half, y - half, x + half, y + half], outline='red')
    return src_img
