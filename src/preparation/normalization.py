#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-11-05'

"""

import time
import os
from skimage import color
import numpy as np
from skimage import io
from core.util import read_csv_file, get_project_root

# Reinhard algorithm
class ImageNormalization(object):
    def __init__(self, method, **kwarg):

        if method == "reinhard":
            self.method = self.normalize_Reinhard
            self.source_mean = kwarg["source_mean"]
            self.source_std = kwarg["source_std"]
            self.target_mean = kwarg["target_mean"]
            self.target_std = kwarg["target_std"]
        elif method == "lab_mean":
            pass
        elif method == "rgb_norm":
            self.method = self.normalize_rgb
            self.source_mean = kwarg["source_mean"]
            self.source_std = kwarg["source_std"]
            self.target_mean = kwarg["target_mean"]
            self.target_std = kwarg["target_std"]
        elif method == "match_hist":
            self.method = self.normalize_hist
            target_path = "{}/data/{}".format(get_project_root(), kwarg["hist_target"])
            hist_target = np.load(target_path).item()

            if kwarg["hist_source"] is not None:
                print("reading histogram file ...")
                source_path = "{}/data/{}".format(get_project_root(), kwarg["hist_source"])
                print("reading histogram file: ", source_path)
                hist_source = np.load(source_path).item()

            else:
                image_source = kwarg["image_source"]
                print("calculating histogram, the number of source: ", len(image_source))
                hist_source = self._calculate_hist(image_source)

            LUT = []
            LUT.append(self._estimate_cumulative_cdf(hist_source["L"], hist_target["L"], start=0, end=100))
            LUT.append(self._estimate_cumulative_cdf(hist_source["A"], hist_target["A"], start=-128, end=127))
            LUT.append(self._estimate_cumulative_cdf(hist_source["B"], hist_target["B"], start=-128, end=127))
            self.LUT = LUT
        return

    def normalize(self, src_img):
       return self.method(src_img)

    def normalize_Reinhard(self, src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        labO_l = np.array(lab_img[:, :, 0])
        labO_a = np.array(lab_img[:, :, 1])
        labO_b = np.array(lab_img[:, :, 2])

        # # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
        labO_l = (labO_l - self.source_mean[0]) / self.source_std[0] * self.target_std[0] + self.target_mean[0]
        labO_a = (labO_a - self.source_mean[1]) / self.source_std[1] * self.target_std[1] + self.target_mean[1]
        labO_b = (labO_b - self.source_mean[2]) / self.source_std[2] * self.target_std[2] + self.target_mean[2]

        labO_l[labO_l > 100] = 100
        labO_l[labO_l < 0] = 0
        labO_a[labO_a > 127] = 127
        labO_a[labO_a < -128] = -128
        labO_b[labO_b > 127] = 127
        labO_b[labO_b < -128] = -128

        labO = np.dstack([labO_l, labO_a, labO_b])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)
        return rgb_image


    def normalize_rgb(self, src_img, ):
        # RGB三通道分离
        rgb_r = src_img[:, :, 0]
        rgb_g = src_img[:, :, 1]
        rgb_b = src_img[:, :, 2]

        rgb1_r= (rgb_r - self.source_mean[0]) / self.source_std[0] * self.target_std[0] + self.target_mean[0]
        rgb1_g = (rgb_g - self.source_mean[1]) / self.source_std[1] * self.target_std[1] + self.target_mean[1]
        rgb1_b = (rgb_b - self.source_mean[2]) / self.source_std[2] * self.target_std[2] + self.target_mean[2]

        rgb1_r[rgb1_r > 255] = 255
        rgb1_r[rgb1_r < 0] = 0
        rgb1_g[rgb1_g > 255] = 255
        rgb1_g[rgb1_g < 0] = 0
        rgb1_b[rgb1_b > 255] = 255
        rgb1_b[rgb1_b < 0] = 0

        rgb_result = np.dstack([rgb1_r.astype(np.int), rgb1_g.astype(np.int), rgb1_b.astype(np.int)])

        return rgb_result

    def _estimate_cumulative_cdf(self, source, template, start, end):
        src_values, src_counts = source
        tmpl_values, tmpl_counts = template

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / np.sum(src_counts)
        tmpl_quantiles = np.cumsum(tmpl_counts) / np.sum(tmpl_counts)

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

        if src_values[0] > start:
            src_values = np.insert(src_values, 0, start)
            interp_a_values = np.insert(interp_a_values, 0, start)
        if src_values[-1] < end:
            src_values = np.append(src_values, end)
            interp_a_values = np.append(interp_a_values, end)

        new_source = np.arange(start, end + 1)
        interp_b_values = np.interp(new_source, src_values, interp_a_values)
        # result = dict(zip(new_source, np.rint(interp_b_values))) # for debug
        # return result
        return np.rint(interp_b_values)

    def _calculate_hist(self, image_list):

        data_L = []
        data_A = []
        data_B = []
        for img in image_list:
            lab_img = color.rgb2lab(img)

            # LAB三通道分离
            labO_l = np.array(lab_img[:, :, 0])
            labO_a = np.array(lab_img[:, :, 1])
            labO_b = np.array(lab_img[:, :, 2])

            data_L.append(labO_l.astype(np.int))
            data_A.append(labO_a.astype(np.int))
            data_B.append(labO_b.astype(np.int))

        data_L = np.array(data_L)
        data_A = np.array(data_A)
        data_B = np.array(data_B)

        L_values, L_counts = np.unique(data_L.ravel(), return_counts=True)
        A_values, A_counts = np.unique(data_A.ravel(), return_counts=True)
        B_values, B_counts = np.unique(data_B.ravel(), return_counts=True)

        return {"L":(L_values, L_counts), "A":(A_values, A_counts), "B":(B_values, B_counts) }

    def normalize_hist(self, src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        lab0_l = np.array(lab_img[:, :, 0]).astype(np.int)
        lab0_a = np.array(lab_img[:, :, 1]).astype(np.int)
        lab0_b = np.array(lab_img[:, :, 2]).astype(np.int)

        LUT_L = self.LUT[0]
        lab1_l = LUT_L[lab0_l]

        LUT_A = self.LUT[1]
        lab1_a = LUT_A[128 + lab0_a]

        LUT_B = self.LUT[2]
        lab1_b = LUT_B[128 + lab0_b]

        labO = np.dstack([lab1_l, lab1_a, lab1_b])
        # LAB to RGB变换, 会除以255
        rgb_image = color.lab2rgb(labO)

        return rgb_image

class ImageNormalizationTool(object):
    def __init__(self, params):
        self._params = params
        # 归一化时，使用的参数
        return

    def calculate_avg_mean_std_RGB(self, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH

        count = 0
        mean_r = []
        mean_g = []
        mean_b = []
        std_r = []
        std_g = []
        std_b = []

        for data_filename in data_filenames:
            data_file = "{}/{}".format(root_path, data_filename)

            f = open(data_file, "r")
            for line in f:
                items = line.split(" ")
                patch_file = "{}/{}".format(root_path, items[0])
                img = io.imread(patch_file, as_gray=False)

                # lab_img = color.rgb2lab(img)

                # RGB三通道分离
                rgb_r = img[:, :, 0]
                rgb_g = img[:, :, 1]
                rgb_b = img[:, :, 2]

                # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
                std_r.append(np.std(rgb_r))
                std_g.append(np.std(rgb_g))
                std_b.append(np.std(rgb_b))

                mean_r.append(np.mean(rgb_r))
                mean_g.append(np.mean(rgb_g))
                mean_b.append(np.mean(rgb_b))

                if (0 == count%1000):
                    print("{} calculate mean and std >>> {}".format(time.asctime( time.localtime()), count))
                count += 1

            f.close()

        avg_mean_r = np.mean(mean_r)
        avg_mean_g = np.mean(mean_g)
        avg_mean_b = np.mean(mean_b)
        avg_std_r = np.mean(std_r)
        avg_std_g = np.mean(std_g)
        avg_std_b = np.mean(std_b)

        return avg_mean_r, avg_mean_g, avg_mean_b, avg_std_r, avg_std_g, avg_std_b


    '''
    Lab颜色空间中的L分量用于表示像素的亮度，取值范围是[0,100],表示从纯黑到纯白；
    a表示从红色到绿色的范围，取值范围是[127,-128]；
    b表示从黄色到蓝色的范围，取值范围是[127,-128]。
    '''
    def calculate_avg_mean_std(self, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH

        count = 0
        mean_l = []
        mean_a = []
        mean_b = []
        std_l = []
        std_a = []
        std_b = []

        for data_filename in data_filenames:
            data_file = "{}/{}".format(root_path, data_filename)

            f = open(data_file, "r")
            for line in f:
                items = line.split(" ")
                patch_file = "{}/{}".format(root_path, items[0])
                img = io.imread(patch_file, as_gray=False)

                lab_img = color.rgb2lab(img)

                # LAB三通道分离
                labO_l = lab_img[:, :, 0]
                labO_a = lab_img[:, :, 1]
                labO_b = lab_img[:, :, 2]

                # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
                std_l.append(np.std(labO_l))
                std_a.append(np.std(labO_a))
                std_b.append(np.std(labO_b))

                mean_l.append(np.mean(labO_l))
                mean_a.append(np.mean(labO_a))
                mean_b.append(np.mean(labO_b))

                if (0 == count%1000):
                    print("{} calculate mean and std >>> {}".format(time.asctime( time.localtime()), count))
                count += 1

            f.close()

        avg_mean_l = np.mean(mean_l)
        avg_mean_a = np.mean(mean_a)
        avg_mean_b = np.mean(mean_b)
        avg_std_l = np.mean(std_l)
        avg_std_a = np.mean(std_a)
        avg_std_b = np.mean(std_b)

        return avg_mean_l, avg_mean_a, avg_mean_b, avg_std_l, avg_std_a, avg_std_b

    def calculate_hist(self, source_code, source_txt, template_code, template_txt):
        root_path = self._params.PATCHS_ROOT_PATH
        print("prepare transform function ...", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        source_path = "{}/{}".format(root_path[source_code], source_txt)
        template_path = "{}/{}".format(root_path[template_code], template_txt)
        source_files, _ = read_csv_file(root_path[source_code], source_path)
        template_files, _ = read_csv_file(root_path[template_code], template_path)
        print("Loaded the number of sources = ", len(source_files), "the number of templates = ", len(template_files))

        project_root = self._params.PROJECT_ROOT

        hist_sources = self._generate_histogram(source_files)
        np.save( project_root + "/data/hist_soures", hist_sources)

        hist_templates = self._generate_histogram(template_files)
        np.save(project_root + "/data/hist_templates", hist_templates)

        return #hist_sources, hist_templates

        # sources = self._loading_hist_data(source_files)
        # templates = self._loading_hist_data(template_files)
        #
        # print("Loaded the number of sources = ", len(sources["L"]), "the number of templates = ", len(templates["L"]))
        #
        # LUT = []
        # LUT.append(self._estimate_cumulative_cdf(sources["L"], templates["L"], start=0, end=100))
        # LUT.append(self._estimate_cumulative_cdf(sources["A"], templates["A"], start=-128, end=127))
        # LUT.append(self._estimate_cumulative_cdf(sources["B"], templates["B"], start=-128, end=127))
        #
        # project_root = self._params.PROJECT_ROOT
        # np.save( project_root + "/data/hist_match_function", LUT)


    # def _loading_hist_data(self, filennames):
    #
    #     results = {"L":[], "A":[], "B": []}
    #
    #     for file in filennames:
    #         img = io.imread(file, as_gray=False)
    #         lab_img = color.rgb2lab(img)
    #
    #         # LAB三通道分离
    #         labO_l = np.array(lab_img[:, :, 0])
    #         labO_a = np.array(lab_img[:, :, 1])
    #         labO_b = np.array(lab_img[:, :, 2])
    #
    #         results["L"].append(labO_l.astype(np.int))
    #         results["A"].append(labO_a.astype(np.int))
    #         results["B"].append(labO_b.astype(np.int))
    #
    #     return results

    # def _estimate_cumulative_cdf(self, source, template, start, end):
    #     source = np.array(source)
    #     template = np.array(template)
    #     src_values, src_counts = np.unique(source.ravel(),  return_counts=True)
    #     tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    #
    #     # calculate normalized quantiles for each array
    #     src_quantiles = np.cumsum(src_counts) / source.size
    #     tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    #
    #     interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    #
    #     if src_values[0] > start:
    #         src_values = np.insert(src_values, 0, start)
    #         interp_a_values = np.insert(interp_a_values, 0, start)
    #     if src_values[-1] < end:
    #         src_values = np.append(src_values, end)
    #         interp_a_values = np.append(interp_a_values, end)
    #
    #     new_source = np.arange(start, end + 1)
    #     interp_b_values = np.interp(new_source, src_values, interp_a_values)
    #     # result = dict(zip(new_source, np.rint(interp_b_values))) # for debug
    #     # return result
    #     return np.rint(interp_b_values)

    def _generate_histogram(self, filennames):
        Shape_L = (101, ) # 100 + 1
        Shape_A = (256, ) # 127 + 128 + 1
        Shape_B = (256, )

        hist_l = np.zeros(Shape_L)
        hist_a = np.zeros(Shape_A)
        hist_b = np.zeros(Shape_B)
        for K, file in enumerate(filennames):
            img = io.imread(file, as_gray=False)
            lab_img = color.rgb2lab(img)

            # LAB三通道分离
            labO_l = np.array(lab_img[:, :, 0])
            labO_a = np.array(lab_img[:, :, 1])
            labO_b = np.array(lab_img[:, :, 2])

            labO_l = np.rint(labO_l)
            labO_a = np.rint(labO_a)
            labO_b = np.rint(labO_b)

            values, counts = np.unique(labO_l.ravel(), return_counts=True)
            for value, count in zip(values, counts):
                hist_l[int(value)] += count

            values, counts = np.unique(labO_a.ravel(), return_counts=True)
            for value, count in zip(values, counts):
                hist_a[int(value) + 128] += count

            values, counts = np.unique(labO_b.ravel(), return_counts=True)
            for value, count in zip(values, counts):
                hist_b[int(value) + 128] += count

            if (0 == K % 1000):
                print("{} calculate histogram >>> {}".format(time.asctime(time.localtime()), K))

        tag = hist_l > 0
        values_l = np.arange(0, 101)
        hist_l = hist_l[tag]
        values_l = values_l[tag]

        tag = hist_a > 0
        values_a = np.arange(-128, 128)
        hist_a = hist_a[tag]
        values_a = values_a[tag]

        tag = hist_b > 0
        values_b = np.arange(-128, 128)
        hist_b = hist_b[tag]
        values_b = values_b[tag]

        return {"L":(values_l, hist_l), "A":(values_a, hist_a), "B":(values_b, hist_b) }
