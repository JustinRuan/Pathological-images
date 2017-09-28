#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import caffe
import numpy as np

deploy = 'D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/DetectCancer/models/lenet/deploy.prototxt'
# 训练好的caffemodel
caffe_model = 'D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/DetectCancer/models/lenet_iter_1000.caffemodel'


def classify(img):
    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network


if __name__ == '__main__':
    classify()
