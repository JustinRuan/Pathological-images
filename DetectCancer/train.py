#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# 设定文件保存路径
solver_proto = 'D:/CloudSpace/DoingNow/WorkSpace/Pathological_Images/DetectCancer/models/lenet/solver.prototxt'


# 开始训练
def training(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()


if __name__ == '__main__':
    training(solver_proto)
