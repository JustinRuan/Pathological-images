#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import utils

# 设定文件保存路径
solver_LeNet_proto = utils.PROJECT_PATH + '/Pathological_Images/DetectCancer/models/lenet/solver.prototxt'
solver_GoogLeNet_proto = utils.PROJECT_PATH + '/Pathological_Images/DetectCancer/models/googlenet/quick_solver.prototxt'

pretrained_model = utils.PROJECT_PATH + '/Pathological_Images/DetectCancer/models/googlenet_iter_600.caffemodel'

# 开始训练
def training(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(solver_proto)
    # 利用snapshot从断点恢复训练
    # solver.net.copy_from(pretrained_model)

    solver.solve()


if __name__ == '__main__':
    training(solver_GoogLeNet_proto)
