#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-28'

"""
from core import Params
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

class cnn_caffe(object):
    # model_name = "googlenet_caffe'
    # samples_name = "ZoneR"
    def __init__(self, params, model_name, samples_name):
        self._params = params
        self.model_name = model_name
        self.model_root = "{}/models/{}/".format(self._params.PROJECT_ROOT, model_name)
        self.train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

        self.train_proto_template = self.model_root + "train_template.prototxt"
        self.train_proto = self.model_root +  "train.prototxt"

        self.solver_proto = self.model_root +  "solver.prototxt"
        self.deploy_proto = self.model_root +  "deploy.prototxt"
        return

    def write_net(self):
        # 读入train_template.prototxt
        with open(self.train_proto_template, 'r') as f1:
            train_temp = f1.read()
            train_temp = train_temp.replace("<train.txt>", self.train_list)
            train_temp = train_temp.replace("<train_folder>", self._params.PATCHS_ROOT_PATH)
            train_temp = train_temp.replace("<test.txt>", self.test_list)
            train_temp = train_temp.replace("<test_folder>", self._params.PATCHS_ROOT_PATH)

        # 写入train.prototxt
        with open(self.train_proto, 'w') as f:
            f.write(train_temp)

        return

    def gen_solver(self, solver_file, train_net, test_net):
        s = caffe_pb2.SolverParameter()
        s.train_net = train_net
        s.test_net.append(test_net)
        s.test_interval = 2400  # 60000/64，测试间隔参数：训练完一次所有的图片，进行一次测试
        s.test_iter.append(800)  # 50000/100 测试迭代次数，需要迭代500次，才完成一次所有数据的测试
        s.max_iter = 2400*3  # 10 epochs , 938*10，最大训练次数
        s.base_lr = 0.001  # 基础学习率
        s.momentum = 0.900  # 动量
        s.weight_decay = 0.0002000  # 权值衰减项

        # s.lr_policy = 'step'  # 学习率变化规则
        # s.stepsize = 3000  # 学习率变化频率
        # s.gamma = 0.1  # 学习率变化指数

        s.lr_policy = 'poly'
        s.power = 0.5

        s.average_loss = 40
        s.display = 20  # 屏幕显示间隔
        s.snapshot = 20000  # 保存caffemodel的间隔
        s.snapshot_prefix = self.model_root + self.model_name  # caffemodel前缀
        s.type = 'SGD'  # 优化算法
        # s.solver_mode = proto.caffe_pb2.SolverParameter.CPU  # 加速
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        # 写入solver.prototxt
        with open(solver_file, 'w') as f:
            f.write(str(s))

    # 开始训练
    def training(self, solver_proto):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        solver = caffe.SGDSolver(solver_proto)
        # 利用snapshot从断点恢复训练
        # solver.net.copy_from(pretrained_model)
        solver.solve()

