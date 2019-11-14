
import random
import unittest

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

from core import *
from pytorch.cancer_map import CancerMapBuilder, SlideFilter
from pytorch.detector import AdaptiveDetector
from pytorch.locator import Locator
from pytorch.slide_predictor import SlidePredictor

JSON_PATH = "D:/CloudSpace/WorkSpace/PatholImage/config/justin2.json"
# JSON_PATH = "E:/Justin/WorkSpace/PatholImage/config/justin_m.json"
# JSON_PATH = "H:/Justin/PatholImage/config/justin3.json"

class TestModel(unittest.TestCase):
    def setUp(self):
        self._params = Params()
        # self._params.load_config_file(JSON_PATH)
        self._params.load_param(GLOBAL_SCALE=1.25, SLICES_ROOT_PATH="D:/Data/CAMELYON16",
                                PATCHS_DICT={"P0619":"D:/Data/Patches/P0619",}, NUM_WORKERS=0)

    def test_detect(self):

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

            imgCone = ImageCone(self._params, Open_Slide())

            # 读取数字全扫描切片图像
            if Train_Tag:
                tag = imgCone.open_slide("Train_Tumor/%s.tif" % slice_id,
                                         'Train_Tumor/%s.xml' % slice_id, slice_id)
            else:
                tag = imgCone.open_slide("Testing/images/%s.tif" % slice_id,
                                         'Testing/images/%s.xml' % slice_id, slice_id)

            detector = AdaptiveDetector(self._params, imgCone)

            if x2 * y2 == 0:
                x1, y1, x2, y2 = detector.get_detection_rectangle()
                print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            history = detector.adaptive_detect(x1, y1, x2, y2, 1.25, extract_scale=40, patch_size=256,
                                               max_iter_nums=20, batch_size=10,
                                               limit_sampling_density=1, enhanced=True,
                                               superpixel_area=1000,
                                               superpixels_boundaries_spacing=120)

            detector.save_result_history(x1, y1, x2, y2, 1.25, history)

            if enable_evaluate:
                cmb = CancerMapBuilder(self._params, extract_scale=40, patch_size=256)
                cancer_map = cmb.generating_probability_map(history, x1, y1, x2, y2, 1.25)

                src_img = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
                mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)

                levels = [0.3, 0.5, 0.8]
                false_positive_rate, true_positive_rate, roc_auc, dice = Evaluation.evaluate_slice_map(cancer_map,
                                                                                                       mask_img,
                                                                                                       levels)
                result[slice_id] = (roc_auc, dice)

                # save results
                # self.show_results(cancer_map, dice, false_positive_rate, history, levels, mask_img, roc_auc, slice_id,
                #                   src_img, true_positive_rate)
                save_path = "{}/results/cancer_pic".format(self._params.PROJECT_ROOT)
                Evaluation.save_result_picture(slice_id, src_img, mask_img, cancer_map, history, roc_auc, levels, save_path)
                # detector.save_result_xml(x1, y1, 1.25, cancer_map, levels)

        for slice, (auc, dices) in result.items():
            print("#################{}###################".format(slice))
            for t, value in dices:
                print("threshold = {:.3f}, dice coef = {:.6f}".format(t, value))
            print("ROC auc: {:.6f}".format(auc))
            print("#################{}###################".format(slice))

    def test_update_history(self):
        sc = SlideFilter(self._params, "Slide_simple", "64")
        sc.update_history(chosen=None, batch_size=100)

    def test_calculate_ROC_pixel_level(self):
        eval = Evaluation(self._params)
        select = ["Tumor_{:0>3d}".format(i) for i in range(1,10)]
        eval.calculate_ROC(None, tag=64, chosen=select, p_thresh=0.5)

    def test_calculate_E1(self):
        sp = SlidePredictor(self._params)

        Tumor_names = ["Tumor_{:0>3d}".format(i) for i in range(1, 112)] # 1, 112
        Normal_names = ["Normal_{:0>3d}".format(i) for i in range(1, 161)] # 1, 161

        dim = 5
        feature_data, label_data = sp.extract_slide_features(tag=0, normal_names=Normal_names,
                                                             tumor_names=Tumor_names, DIM=dim)

        rand = random.randint(1, 100)
        print("rand", rand)
        X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2,
                                                            random_state=rand)

        max_iter = 10000

        #'kernel': ('linear'),
        parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 1.2, 1.5, 2.0, 10]}

        svc = svm.SVC(kernel='linear', probability=True,max_iter=max_iter,verbose=0,)
        grid  = GridSearchCV(svc, parameters, cv=5)
        grid .fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        clf = grid.best_estimator_
        sp.save_model(clf, "linearsvm", X_train, y_train, X_test, y_test)

    def test_calculate_E2(self):
        loca = Locator(self._params)
        select = ["Tumor_{:0>3d}".format(i) for i in range(1,112)]
        loca.output_result_csv("csv_0", tag =64,  chosen=select)

        eval = Evaluation(self._params)

        mask_folder = "{}/data/true_masks".format(self._params.PROJECT_ROOT)
        result_folder = "{}/results/csv_0".format(self._params.PROJECT_ROOT)

        eval.evaluation_FROC(mask_folder, result_folder, level = 7)



