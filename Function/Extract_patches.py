from patches import get_roi, get_seeds
import utils

class Patch(object):
    def __init__(self):
        self.seeds = []

    def get_roi_seeds(self, src_image, distance):
        roi_img = get_roi(src_image)
        self.seeds = get_seeds(roi_img, distance)
        return

    # x is col, y is row
    def detect_cancer_patch(self, mask_img, x, y, patch_width):
        half = int(patch_width / 2)
        sub_m = mask_img[y - half: y + half, x - half: x + half]
        total = sub_m.sum()
        r = total / (patch_width * patch_width)
        return r > 0.85

    def extract_patches(self, slide, mask_img, scale, patch_width):
        i = 10000000
        for (x, y) in self.seeds:
            isCancer = self.detect_cancer_patch(mask_img, x, y, utils.PATCH_SIZE_LOW)
            xx = int(utils.AMPLIFICATION_SCALE * x)
            yy = int(utils.AMPLIFICATION_SCALE * y)

            patch_data = slide.get_image_block(scale, xx, yy, patch_width, patch_width, True)
            if isCancer:
                filename = str(i) + '.jpg'
                i += 1
                patch_data.tofile(utils.PATCH_PATH_CANCER + filename)
            else:
                filename = str(i) + '.jpg'
                i += 1
                patch_data.tofile(utils.PATCH_PATH_NORMAL + filename)
        return
