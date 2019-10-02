from core import *
import matplotlib.pyplot as plt
from pytorch.detector import Detector, AdaptiveDetector
import numpy as np
from skimage.segmentation import mark_boundaries
from pytorch.cancer_map import CancerMapBuilder

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

if __name__ == "__main__":

    slide_list = [1]
    result = {}
    Train_Tag = True

    # 如果输出癌变概率图，并进行评估
    enable_evaluate = False

    for id in slide_list:
        x1, y1, x2, y2 = 0, 0, 0, 0

        if Train_Tag:
            slice_id = "Tumor_{:0>3d}".format(id)
        else:
            slice_id = "Test_{:0>3d}".format(id)

        c = Params()
        c.load_config_file(JSON_PATH)
        imgCone = ImageCone(c, Open_Slide())

        # 读取数字全扫描切片图像
        if Train_Tag:
            tag = imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                                     'Train_Tumor/%s.xml' % slice_id, slice_id)
        else:
            tag = imgCone.open_slide("Testing/images/%s.tif" % slice_id,
                                     'Testing/images/%s.xml' % slice_id, slice_id)

        detector = AdaptiveDetector(c, imgCone)

        if x2 * y2 == 0:
            x1, y1, x2, y2 = detector.get_detection_rectangle()
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        history = detector.adaptive_detect(x1, y1, x2, y2, 1.25, extract_scale=40, patch_size=256,
                                           max_iter_nums=20, batch_size=10,
                                           limit_sampling_density=2, enhanced=True,
                                           superpixel_area=1000,
                                           superpixels_boundaries_spacing=60)

        detector.save_result_history(x1, y1, x2, y2, 1.25, history)

        if enable_evaluate:
            cmb = CancerMapBuilder(c, extract_scale=40, patch_size=256)
            cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

            src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
            mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

            levels = [0.3, 0.5, 0.8]
            false_positive_rate, true_positive_rate, roc_auc, dice = Evaluation.evaluate_slice_map(cancer_map,
                                                                                                   mask_img,
                                                                                                   levels)
            result[slice_id] = (roc_auc, dice)

            # 存盘输出部分
            # self.show_results(cancer_map, dice, false_positive_rate, history, levels, mask_img, roc_auc, slice_id,
            #                   src_img, true_positive_rate)
            save_path = "{}/results/cancer_pic".format(c.PROJECT_ROOT)
            Evaluation.save_result_picture(slice_id, src_img, mask_img, cancer_map, history, roc_auc, levels, save_path)
            # detector.save_result_xml(x1, y1, 1.25, cancer_map, levels)

    for slice, (auc, dices) in result.items():
        print("#################{}###################".format(slice))
        for t, value in dices:
            print("threshold = {:.3f}, dice coef = {:.6f}".format(t, value))
        print("ROC auc: {:.6f}".format(auc))
        print("#################{}###################".format(slice))


