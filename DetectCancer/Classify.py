#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import caffe
from patches import DigitalSlide, Patch, get_roi, get_seeds, draw_seeds
import utils
import numpy as np
from matplotlib import pyplot as plt

deploy = 'D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/DetectCancer/models/lenet/deploy.prototxt'
# 训练好的caffemodel
caffe_model = 'D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/DetectCancer/models/lenet_iter_1000.caffemodel'


# def classify(net, img):
#
#
#     return 0.1

def classify_slide():
    slide = DigitalSlide()
    tag = slide.open_slide("D:/Study/breast/3Plus/17004930 HE_2017-07-29 09_45_09.kfb", "17004930")

    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network
    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

    result = []
    space_patch = utils.PATCH_SIZE_LOW

    if tag:
        ImageWidth, ImageHeight = slide.get_image_width_height_byScale(utils.GLOBAL_SCALE)
        fullImage = slide.get_image_block(utils.GLOBAL_SCALE, 0, 0, ImageWidth, ImageHeight)
        result = np.zeros(
            (np.rint(ImageHeight / space_patch).astype(np.int32), np.rint(ImageWidth / space_patch).astype(np.int32)),
            dtype=np.uint8)

        roi_img = get_roi(fullImage)
        seeds = get_seeds(roi_img, 1)

        # 低分辨率上的间隔为PATCH_SIZE_LOW 的坐标点
        for (x, y) in seeds:
            xx = int(utils.AMPLIFICATION_SCALE * x)  # 高分辨率上对应的坐标点
            yy = int(utils.AMPLIFICATION_SCALE * y)
            patch_img = slide.get_image_block(utils.EXTRACT_SCALE, xx, yy, utils.PATCH_SIZE_HIGH, utils.PATCH_SIZE_HIGH,
                                              False)
            patch = np.array(patch_img)
            net.blobs['data'].data[...] = transformer.preprocess('data', patch)
            # 执行测试
            net.forward()
            prob = net.blobs['prob'].data[0].flatten()
            if prob[1] > 0:  # cancer
                mx = (np.rint(x / space_patch)).astype(np.int32)
                my = (np.rint(y / space_patch)).astype(np.int32)
                result[my, mx] = 1

    tag = slide.release_slide_pointer()
    return fullImage, result

if __name__ == '__main__':
    full_img, result_img = classify_slide()
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    ax = axes.ravel()

    ax[0].imshow(full_img)
    ax[0].set_title("full_img")

    ax[1].imshow(result_img)
    ax[1].set_title("result_img")

    for a in ax.ravel():
        a.axis('off')

    plt.show()

    result_img.tofile("he_result_img.bin")
