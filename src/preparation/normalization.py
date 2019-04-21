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

from core.util import read_csv_file, get_project_root, get_seeds
from preparation.hsd_transform import hsd2rgb, rgb2hsd
from visdom import Visdom
from core import Random_Gen

class AbstractNormalization(object):
    def __init__(self, method, **kwarg):
        self.method_name = method

    def process(self, src_img):
        return self.normalize(src_img)

    def normalize(self, src_img):
       raise NotImplementedError

    def normalize_on_batch(self, src_img_list):
        result = []
        for img in src_img_list:
            result.append(self.normalize(img))

        return result

class RGBNormalization(AbstractNormalization):
    def __init__(self, method, **kwarg):
        super(RGBNormalization, self).__init__(method, **kwarg)

        self.source_mean = kwarg["source_mean"]
        self.source_std = kwarg["source_std"]
        self.target_mean = kwarg["target_mean"]
        self.target_std = kwarg["target_std"]

    def normalize(self, src_img):
        # RGB三通道分离
        rgb_r = src_img[:, :, 0]
        rgb_g = src_img[:, :, 1]
        rgb_b = src_img[:, :, 2]

        rgb1_r= (rgb_r - self.source_mean[0]) / self.source_std[0] * self.target_std[0] + self.target_mean[0]
        rgb1_g = (rgb_g - self.source_mean[1]) / self.source_std[1] * self.target_std[1] + self.target_mean[1]
        rgb1_b = (rgb_b - self.source_mean[2]) / self.source_std[2] * self.target_std[2] + self.target_mean[2]

        # rgb1_r[rgb1_r > 255] = 255
        # rgb1_r[rgb1_r < 0] = 0
        # rgb1_g[rgb1_g > 255] = 255
        # rgb1_g[rgb1_g < 0] = 0
        # rgb1_b[rgb1_b > 255] = 255
        # rgb1_b[rgb1_b < 0] = 0

        rgb1_r = np.clip(rgb1_r, 0, 255)
        rgb1_g = np.clip(rgb1_g, 0, 255)
        rgb1_b = np.clip(rgb1_b, 0, 255)

        rgb_result = np.dstack([rgb1_r.astype(np.int), rgb1_g.astype(np.int), rgb1_b.astype(np.int)])

        return rgb_result

# Reinhard algorithm
class ReinhardNormalization(AbstractNormalization):
    def __init__(self, method, **kwarg):
        super(ReinhardNormalization, self).__init__(method, **kwarg)

        self.source_mean = kwarg["source_mean"]
        self.source_std = kwarg["source_std"]
        self.target_mean = kwarg["target_mean"]
        self.target_std = kwarg["target_std"]

    def normalize(self, src_img):
        lab_img = color.rgb2lab(src_img)

        # LAB三通道分离
        labO_l = np.array(lab_img[:, :, 0])
        labO_a = np.array(lab_img[:, :, 1])
        labO_b = np.array(lab_img[:, :, 2])

        # # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
        labO_l = (labO_l - self.source_mean[0]) / self.source_std[0] * self.target_std[0] + self.target_mean[0]
        labO_a = (labO_a - self.source_mean[1]) / self.source_std[1] * self.target_std[1] + self.target_mean[1]
        labO_b = (labO_b - self.source_mean[2]) / self.source_std[2] * self.target_std[2] + self.target_mean[2]

        # labO_l[labO_l > 100] = 100
        # labO_l[labO_l < 0] = 0
        # labO_a[labO_a > 127] = 127
        # labO_a[labO_a < -128] = -128
        # labO_b[labO_b > 127] = 127
        # labO_b[labO_b < -128] = -128

        labO_l = np.clip(labO_l, 0, 100)
        labO_a = np.clip(labO_a, -128, 127)
        labO_b = np.clip(labO_b, -128, 127)

        labO = np.dstack([labO_l, labO_a, labO_b])
        # LAB to RGB变换
        rgb_image = color.lab2rgb(labO)
        return rgb_image

class HSDNormalization(AbstractNormalization):
    def __init__(self, method, **kwarg):
        super(HSDNormalization, self).__init__(method, **kwarg)

        self.source_mean = kwarg["source_mean"]
        self.source_std = kwarg["source_std"]
        self.target_mean = kwarg["target_mean"]
        self.target_std = kwarg["target_std"]

    def normalize(self, src_img):
        hsd_img = rgb2hsd(src_img)

        # LAB三通道分离
        hsdO_h = hsd_img[:, :, 0]
        hsdO_s = hsd_img[:, :, 1]
        hsdO_d = hsd_img[:, :, 2]

        # # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
        hsdO_h = (hsdO_h - self.source_mean[0]) / self.source_std[0] * self.target_std[0] + self.target_mean[0]
        hsdO_s = (hsdO_s - self.source_mean[1]) / self.source_std[1] * self.target_std[1] + self.target_mean[1]
        hsdO_d = (hsdO_d - self.source_mean[2]) / self.source_std[2] * self.target_std[2] + self.target_mean[2]

        hsd1 = np.dstack([hsdO_h, hsdO_s, hsdO_d])
        # LAB to RGB变换
        rgb_image = hsd2rgb(hsd1)
        return rgb_image

class HistNormalization(AbstractNormalization):
    def __init__(self, method, **kwarg):
        super(HistNormalization, self).__init__(method, **kwarg)

        target_path = "{}/data/{}".format(get_project_root(), kwarg["hist_target"])
        hist_target = np.load(target_path).item()
        self.hist_target = hist_target

        self._history = []
        self.enable_update = True

        if kwarg["hist_source"] is not None:
            print("reading histogram file ...")
            source_path = "{}/data/{}".format(get_project_root(), kwarg["hist_source"])
            print("reading histogram file: ", source_path)
            hist_source = np.load(source_path).item()

            LUT = []
            LUT.append(self._estimate_cumulative_cdf(hist_source["L"], hist_target["L"], start=0, end=100))
            LUT.append(self._estimate_cumulative_cdf(hist_source["A"], hist_target["A"], start=-128, end=127))
            LUT.append(self._estimate_cumulative_cdf(hist_source["B"], hist_target["B"], start=-128, end=127))
            self.LUT = LUT
            self.hist_source = hist_source
            self.hist_target = hist_target
        else:
            # 将使用Prepare过程进行初始化
            self.LUT = None
            self.hist_source = None

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

    def normalize(self, src_img):
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

    def prepare(self, image_list):
        if not self.enable_update:
            return

        # print("calculating histogram, the number of source: ", len(image_list))
        hist_source = self._calculate_hist(image_list)
        # source_path = "{}/data/{}".format(get_project_root(), "hist_source_tmp")
        # np.save(source_path, hist_source)
        hist_target = self.hist_target
        LUT = []
        LUT.append(self._estimate_cumulative_cdf(hist_source["L"], hist_target["L"], start=0, end=100))
        LUT.append(self._estimate_cumulative_cdf(hist_source["A"], hist_target["A"], start=-128, end=127))
        LUT.append(self._estimate_cumulative_cdf(hist_source["B"], hist_target["B"], start=-128, end=127))
        # update
        self.LUT = LUT
        self.hist_source = hist_source

    def draw_hist(self,fig_name):
        hist_source = self.hist_source
        hist_target = self.hist_target

        viz = Visdom(env="main")
        pic_L = viz.line(
            Y=hist_target["L"][1],
            X=hist_target["L"][0],
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'xlabel': 'L channel',
                'ylabel': 'Probability',
                'title': 'Histogram of L - {}'.format(fig_name),
            },
            name='target',

        )

        viz.line(
            Y=hist_source["L"][1],
            X=hist_source["L"][0],
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
            },
            name='source',
            win=pic_L,
            update='insert',
        )

        pic_A = viz.line(
            Y=hist_target["A"][1],
            X=hist_target["A"][0],
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'xlabel': 'A channel',
                'ylabel': 'Probability',
                'title': 'Histogram of A - {}'.format(fig_name),
            },
            name='target',

        )

        viz.line(
            Y=hist_source["A"][1],
            X=hist_source["A"][0],
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
            },
            name='source',
            win=pic_A,
            update='insert',
        )

        pic_B = viz.line(
            Y=hist_target["B"][1],
            X=hist_target["B"][0],
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'xlabel': 'B channel',
                'ylabel': 'Probability',
                'title': 'Histogram of B - {}'.format(fig_name),
            },
            name='target',

        )

        viz.line(
            Y=hist_source["B"][1],
            X=hist_source["B"][0],
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
            },
            name='source',
            win=pic_B,
            update='insert',
        )

    def draw_normalization_func(self, fig_name):

        viz = Visdom(env="main")
        pic_func = viz.line(
            Y=self.LUT[0],
            X=np.arange(0, 101),
            opts={
                'linecolor': np.array([
                    [0, 0, 255],
                ]),
                'dash': np.array(['solid']),  # 'solid', 'dash', 'dashdot'
                'showlegend': True,
                'xlabel': 'range',
                'ylabel': 'value',
                'title': 'function -{}'.format(fig_name),
            },
            name='L',

        )

        viz.line(
            Y=self.LUT[1],
            X=np.arange(-128, 128),
            opts={
                'linecolor': np.array([
                    [0, 255, 0],
                ]),
            },
            name='A',
            win=pic_func,
            update='insert',
        )

        viz.line(
            Y=self.LUT[2],
            X=np.arange(-128, 128),
            opts={
                'linecolor': np.array([
                    [255, 0, 0],
                ]),
            },
            name='B',
            win=pic_func,
            update='insert',
        )

    @staticmethod
    def get_normalization_function(imgCone, params, extract_scale, patch_size, ):
        low_scale = params.GLOBAL_SCALE
        # 在有效检测区域内，均匀抽样
        eff_region = imgCone.get_effective_zone(low_scale)
        sampling_interval = 1000
        seeds = get_seeds(eff_region, low_scale, extract_scale, patch_size, spacingHigh=sampling_interval, margin=-4)

        # #不受限制地随机抽样
        # rx2 = int(imgCone.ImageWidth * extract_scale / params.GLOBAL_SCALE)
        # ry2 = int(imgCone.ImageHeight * extract_scale / params.GLOBAL_SCALE)
        # random_gen = Random_Gen("halton")
        #
        # N = 2000
        # # rx1, ry1, rx2, ry2 = self.valid_rect
        # x, y = self.random_gen.generate_random(N, 0, rx2, 0, ry2)

        images = []
        for x, y in seeds:
            block = imgCone.get_image_block(extract_scale, x, y, patch_size, patch_size)
            img = block.get_img()
            images.append(img)

        normal = HistNormalization("match_hist", hist_target ="hist_templates.npy",
                                   hist_source = None)
        normal.prepare(images)

        return normal

class ImageNormalizationTool(object):
    def __init__(self, params):
        self._params = params
        # 归一化时，使用的参数
        return

    def calculate_avg_mean_std_RGB(self, source_code, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH[source_code]

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
    def calculate_avg_mean_std(self, source_code, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH[source_code]

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

    def calculate_hist(self, source_code, source_txt, file_code):
        def _generate_histogram(filennames):
            Shape_L = (101,)  # 100 + 1
            Shape_A = (256,)  # 127 + 128 + 1
            Shape_B = (256,)

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

            return {"L": (values_l, hist_l), "A": (values_a, hist_a), "B": (values_b, hist_b)}

        root_path = self._params.PATCHS_ROOT_PATH
        print("prepare transform function ...", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        source_path = "{}/{}".format(root_path[source_code], source_txt)
        source_files, _ = read_csv_file(root_path[source_code], source_path)
        print("Loaded the number of images = ", len(source_files))
        hist_sources = _generate_histogram(source_files)

        project_root = self._params.PROJECT_ROOT
        np.save("{}/data/{}".format(project_root, file_code), hist_sources)
        return


    def calculate_avg_mean_std_HSD(self, source_code, data_filenames):
        root_path = self._params.PATCHS_ROOT_PATH[source_code]

        count = 0
        mean_h = []
        mean_s = []
        mean_d = []
        std_h = []
        std_s = []
        std_d = []

        for data_filename in data_filenames:
            data_file = "{}/{}".format(root_path, data_filename)

            f = open(data_file, "r")
            for line in f:
                items = line.split(" ")
                patch_file = "{}/{}".format(root_path, items[0])
                img = io.imread(patch_file, as_gray=False)

                hsd_img = rgb2hsd(img)

                # HSD三通道分离
                hsdO_h = hsd_img[:, :, 0]
                hsdO_s = hsd_img[:, :, 1]
                hsdO_d = hsd_img[:, :, 2]

                # 按通道进行归一化整个图像, 经过缩放后的数据具有零均值以及标准方差
                std_h.append(np.std(hsdO_h))
                std_s.append(np.std(hsdO_s))
                std_d.append(np.std(hsdO_d))

                mean_h.append(np.mean(hsdO_h))
                mean_s.append(np.mean(hsdO_s))
                mean_d.append(np.mean(hsdO_d))

                if (0 == count%1000):
                    print("{} calculate mean and std >>> {}".format(time.asctime( time.localtime()), count))
                count += 1

            f.close()

        avg_mean_h = np.mean(mean_h)
        avg_mean_s = np.mean(mean_s)
        avg_mean_d = np.mean(mean_d)
        avg_std_h = np.mean(std_h)
        avg_std_s = np.mean(std_s)
        avg_std_d = np.mean(std_d)

        return avg_mean_h, avg_mean_s, avg_mean_d, avg_std_h, avg_std_s, avg_std_d
