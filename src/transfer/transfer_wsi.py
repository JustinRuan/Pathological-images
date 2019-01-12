from core import *
import matplotlib.pyplot as plt
from cnn import *
import numpy as np
import cv2
from transfer import *
import random
from scipy.interpolate import griddata

N = 500
CONFIG_PATH = r'D:\code\python\Pathological-images\config\myConfig.json'


def get_high_seeds(lowScale, highScale, patch_size, low_seed):
    amp = highScale / lowScale
    spacingHigh = patch_size / 2
    space_patch = spacingHigh / amp
    x = (np.rint(low_seed.T[0] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # col
    y = (np.rint(low_seed.T[1] / space_patch + 0.5) * spacingHigh).astype(np.int32)  # row
    resultHigh = []
    for xx, yy in zip(x, y):
        resultHigh.append((xx, yy))
    return np.array(resultHigh)


def get_random_seed(n, x1, x2, y1, y2, sobel_img):
    seed1 = []
    seed2 = []
    if sobel_img is not None:
        probs = []
        for i in range(n):
            x = np.ceil(random.uniform(x1, x2-1)).astype('int64')
            y = np.ceil(random.uniform(y1, y2-1)).astype('int64')
            prob = sobel_img[y - y1, x - x1]
            probs.append((x, y, abs(prob)))
        probs.sort(key=lambda axis: axis[2], reverse=True)
        probs = probs[:N]
        for x, y, prob in probs:
            seed1.append((x - x1, y - y1))
            seed2.append((x, y))
    else:
        for i in range(n):
            x = np.ceil(random.uniform(x1, x2-1)).astype('int64')
            y = np.ceil(random.uniform(y1, y2-1)).astype('int64')
            seed1.append((x - x1, y - y1))
            seed2.append((x, y))
    seed1 = np.array(seed1)
    seed2 = np.array(seed2)
    return seed1, seed2


def inter_sobel(point, value, grid_x_y, threshold, method='nearest', fill_value=0.0):
    for x in value:
        if (x[0] == 0.0) & (x[1] > threshold):
            x[1] = 0
        elif x[1] < threshold:
            x[1] = 0

    value = np.array(value)
    interpolate = griddata(point, value[:, 1], grid_x_y, method=method, fill_value=fill_value)
    sobel_img = cv2.Sobel(interpolate, -1, 2, 2)
    return interpolate, sobel_img


def load_model(params, scale):
    cnn = Transfer(params, "densenet121", "500_128")
    model_path = "{}/models/{}".format(params.PROJECT_ROOT, "densenet121_S{}_merge_cnn.h5".format(scale))
    model = cnn.load_model(mode=999, model_file=model_path)

    model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn, model


def predict(cnn, model, imageCone, high_seed, scale, patch_size, batch=100):
    predictions = cnn.predict_on_batch(model, imageCone, scale, patch_size, high_seed, batch)
    return np.array(predictions)


def run(scale, patch_size, iter_nums, t, id):
    params = Params()
    params.load_config_file(CONFIG_PATH)
    openslide = Open_Slide()
    # openslide2 = KFB_Slide(params.KFB_SDK_PATH)
    # image2 = ImageCone(params, openslide2)
    image = ImageCone(params, openslide)
    # 全切片范围
    test_set = {
                "001": (0, 1679, 2892, 5197),
                "044": (410, 2895, 2813, 6019),
                "047": (391, 2402, 2891, 4280),
                "003": (721, 3244, 3044, 5851),
                }
    roi = test_set[id]
    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]
    image.open_slide("D:/code/dataset/camelyon16/Tumor_%s.tif" % id,
                     'D:/code/dataset/camelyon16/tumor_%s.xml' % id, "Tumor_%s" % id)
    detector = Detector(params, image)
    detector.setting_detected_area(x1, y1, x2, y2, 1.25)
    roc_image = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
    mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
    width = x2 - x1
    high = y2 - y1
    # 生成坐标网格
    grid_y, grid_x = np.mgrid[0: high: 1, 0: width: 1]
    cancer_map = None
    prob_map = None
    count_map = None
    sobel_img = None
    cnn, model = load_model(params, scale)
    all_seed = None
    interpolate_img = None
    all_predictions = []
    for i in range(iter_nums):
        print("iter %d" % (i+1))
        if i == 0:
            points, seed = get_random_seed(N, x1, x2, y1, y2, sobel_img)
            all_seed = points.tolist()
        else:
            points, seed = get_random_seed(2 * N, x1, x2, y1, y2, sobel_img)
            all_seed = all_seed + points.tolist()

        high_seed = get_high_seeds(1.25, scale, patch_size, seed)
        predictions = predict(cnn, model, image, high_seed, scale, patch_size)
        all_predictions = all_predictions + predictions.tolist()
        cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, scale, 1.25, high_seed, predictions, patch_size,
                                                                     prob_map, count_map)
        new_seed = np.array(all_seed)
        interpolate_img, sobel_img = inter_sobel(new_seed, all_predictions, (grid_x, grid_y), t, method='cubic')

    false_positive_rate, true_positive_rate, roc_auc = detector.evaluate(t, interpolate_img, mask_img)

    np.savez("./data/{}_{}_image".format(id, t), false_positive_rate, true_positive_rate,
             roc_auc, roc_image, mask_img, cancer_map, interpolate_img)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=100)
    ax = axes.ravel()
    ax[0].set_title('Receiver Operating Characteristic')
    ax[0].plot(false_positive_rate, true_positive_rate, 'g',
               label='x5  AUC = %0.2f' % roc_auc)
    ax[0].legend(loc='lower right')
    ax[0].plot([0, 1], [0, 1], 'r--')
    ax[0].set_xlim([-0.1, 1.2])
    ax[0].set_ylim([-0.1, 1.2])
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_xlabel('False Positive Rate')

    ax[1].imshow(roc_image)
    ax[1].contour(mask_img, [0.5], linewidths=0.5, colors='r')
    ax[2].imshow(cancer_map)
    ax[3].imshow(mask_img)
    ax[4].imshow(interpolate_img)
    ax[5].imshow(mask_img)
    ax[5].imshow(interpolate_img > t, alpha=0.5, cmap='Blues')
    for a in ax.ravel():
        a.axis('off')
    plt.show()


def draw():
    data1 = np.load("./data/{}_{}_image.npz".format('044', 0.5))
    data2 = np.load('./data/{}_{}_image.npz'.format('047', 0.5))

    false_positive_rate1 = data1['arr_0']
    true_positive_rate1 = data1['arr_1']
    roc_auc1 = data1['arr_2']
    roc_image1 = data1['arr_3']
    mask_img1 = data1['arr_4']
    cancer_map1 = data1['arr_5']
    b1 = data1['arr_6']

    false_positive_rate2 = data2['arr_0']
    true_positive_rate2 = data2['arr_1']
    roc_auc2 = data2['arr_2']
    roc_image2 = data2['arr_3']
    mask_img2 = data2['arr_4']
    cancer_map2 = data2['arr_5']
    b2 = data2['arr_6']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=100)

    ax = axes.ravel()
    ax[0].imshow(roc_image1)
    ax[0].set_title('A')
    ax[1].imshow(roc_image1)
    ax[1].contour(mask_img1, [0], linewidths=1, colors='b')
    ax[1].set_title('B')
    ax[2].imshow(roc_image1)
    b1[b1 < 0] = None
    b1[b1 < 0.5] = None
    ax[2].imshow(b1, cmap='Oranges_r', alpha=1)
    ax[2].set_title('C')

    ax = axes.ravel()
    ax[3].imshow(roc_image2)
    ax[3].set_title('D')
    ax[4].imshow(roc_image2)
    ax[4].contour(mask_img2, [0], linewidths=1, colors='b')
    ax[4].set_title('E')
    ax[5].imshow(roc_image2)
    b2[b2 < 0] = None
    b2[b2 < 0.5] = None
    ax[5].imshow(b2, cmap='Oranges_r', alpha=1)
    ax[5].set_title('F')

    for a in ax.ravel():
        a.axis('off')
    plt.show()


# 手动获取全切片有效范围
def get_full_image(id, scale=1.25):
    params = Params()
    params.load_config_file(CONFIG_PATH)
    openslide = Open_Slide()
    image = ImageCone(params, openslide)
    test_set = {"001": (0, 1679, 2892, 5197),
                "003": (721, 3244, 3044, 5851),
                "044": (410, 2895, 2813, 6019),
                "047": (391, 2402, 2891, 4280),
                }
    id = "001"
    roi = test_set[id]
    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]
    image.open_slide("D:/code/dataset/camelyon16/Tumor_%s.tif" % id,
                     'D:/code/dataset/camelyon16/tumor_%s.xml' % id, "Tumor_%s" % id)
    full_image = image.get_fullimage_byScale(scale)
    detector = Detector(params, image)
    detector.setting_detected_area(x1, y1, x2, y2, 1.25)
    roc_image = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
    plt.imshow(full_image)
    plt.show()
    plt.imshow(roc_image)
    plt.show()


# 肿瘤医院切片测试
# def run2(scale, patch_size, iter_nums, t):
#     params = Params()
#     params.load_config_file(CONFIG_PATH)
#     # openslide = Open_Slide()
#     openslide = KFB_Slide(params.KFB_SDK_PATH)
#     image = ImageCone(params, openslide)
#     # image = ImageCone(params, openslide)
#     # 全切片范围
#     roi = (570, 495, 2244, 1798)
#     x1 = roi[0]
#     y1 = roi[1]
#     x2 = roi[2]
#     y2 = roi[3]
#     image.open_slide("D:/code/python/Pathological-images/data/origin_data/17004930 HE_2017-07-29 09_45_09.kfb",
#                      "D:/code/python/Pathological-images/data/origin_data/17004930 HE_2017-07-29 09_45_09.kfb.Ano",
#                      "17004930")
#     detector = Detector(params, image)
#     detector.setting_detected_area(x1, y1, x2, y2, 1.25)
#     roc_image = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
#     mask_img = detector.get_true_mask_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
#     width = x2 - x1
#     high = y2 - y1
#     # 生成坐标网格
#     grid_y, grid_x = np.mgrid[0: high: 1, 0: width: 1]
#     cancer_map = None
#     prob_map = None
#     count_map = None
#     sobel_img = None
#     cnn, model = load_model(params, scale)
#     all_seed = None
#     b = None
#     all_predictions = []
#     for i in range(iter_nums):
#         print("iter %d" % (i + 1))
#         if i == 0:
#             points, seed = get_random_seed(N, x1, x2, y1, y2, sobel_img)
#             all_seed = points.tolist()
#         else:
#             points, seed = get_random_seed(2 * N, x1, x2, y1, y2, sobel_img)
#             all_seed = all_seed + points.tolist()
#
#         high_seed = get_high_seeds(1.25, scale, patch_size, seed)
#         predictions = predict(cnn, model, image, high_seed, scale, patch_size)
#         all_predictions = all_predictions + predictions.tolist()
#         cancer_map, prob_map, count_map = detector.create_cancer_map(x1, y1, 1.25, scale, 1.25, high_seed,
#                                                                      predictions, patch_size,
#                                                                      prob_map, count_map)
#         new_seed = np.array(all_seed)
#         b, sobel_img = inter_sobel(new_seed, all_predictions, (grid_x, grid_y), t, method='cubic')
#
#     false_positive_rate, true_positive_rate, roc_auc = detector.evaluate(t, b, mask_img)
#
#     fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=100)
#     ax = axes.ravel()
#     ax[0].set_title('Receiver Operating Characteristic')
#     ax[0].plot(false_positive_rate, true_positive_rate, 'g',
#                label='x5  AUC = %0.2f' % roc_auc)
#     ax[0].legend(loc='lower right')
#     ax[0].plot([0, 1], [0, 1], 'r--')
#     ax[0].set_xlim([-0.1, 1.2])
#     ax[0].set_ylim([-0.1, 1.2])
#     ax[0].set_ylabel('True Positive Rate')
#     ax[0].set_xlabel('False Positive Rate')
#
#     ax[1].imshow(roc_image)
#     ax[2].imshow(cancer_map)
#     ax[3].imshow(mask_img)
#     ax[4].imshow(b)
#     ax[5].imshow(mask_img)
#     ax[5].imshow(b > t, alpha=0.5, cmap='Blues')
#     for a in ax.ravel():
#         a.axis('off')
#     plt.show()
#
#     # plt.imshow(b)
#     # plt.plot([seed[0] for seed in all_seed], [seed[1] for seed in all_seed],
#     #          '.', label='Data', color='black', alpha=0.6)
#     # plt.show()
#
#
# # 手动获取全切片有效范围
# def get_full_image2(scale=1.25):
#     params = Params()
#     params.load_config_file(CONFIG_PATH)
#     roi = (570, 495, 2244, 1798)
#     x1 = roi[0]
#     y1 = roi[1]
#     x2 = roi[2]
#     y2 = roi[3]
#
#     openslide = KFB_Slide(params.KFB_SDK_PATH)
#     image = ImageCone(params, openslide)
#     # image = ImageCone(params, openslide)
#
#     image.open_slide("D:/code/python/Pathological-images/data/origin_data/17004930 HE_2017-07-29 09_45_09.kfb",
#                      "D:/code/python/Pathological-images/data/origin_data/17004930 HE_2017-07-29 09_45_09.kfb.Ano",
#                      "17004930")
#     full_image = image.get_fullimage_byScale(scale)
#     detector = Detector(params, image)
#     detector.setting_detected_area(x1, y1, x2, y2, 1.25)
#     roc_image = detector.get_img_in_detect_area(x1, y1, x2, y2, 1.25, 1.25)
#     plt.imshow(full_image)
#     plt.show()
#     plt.imshow(roc_image)
#     plt.show()


if __name__ == '__main__':
    ids = ['001', '003', '044', '047']
    run(scale=20, patch_size=256, iter_nums=10, t=0.5, id=ids[3])
    # get_full_image(ids[0])
    # draw()

    # run2(scale=20, patch_size=256, iter_nums=5, t=0.5)
    # get_full_image2()
