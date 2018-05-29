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
from skimage import io
import numpy as np
from sklearn import metrics

class cnn_caffe(object):
    # model_name = "googlenet_caffe'
    # samples_name = "ZoneR"
    def __init__(self, params, model_name, samples_name):
        self._params = params
        self.model_name = model_name
        self.model_root = "{}/models/{}/".format(self._params.PROJECT_ROOT, model_name)
        self.train_list = "{}/{}_train.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.test_list = "{}/{}_test.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        self.check_list = "{}/{}_check.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)

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

    def testing(self, caffe_model):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy_proto, caffe_model, caffe.TEST)

        # 设定图片的shape格式(1,3,28,28)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
        transformer.set_transpose('data', (2, 0, 1))
        # 减去均值，若训练时未用到均值文件，则不需要此步骤
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        # 缩放到【0，255】之间
        transformer.set_raw_scale('data', 255)
        # 交换通道，将图片由RGB变为BGR
        transformer.set_channel_swap('data', (2, 1, 0))

        expected_tags = []
        predicted_tags = []

        f = open(self.check_list, "r")
        for line in f:
            items = line.split(" ")

            tag = int(items[1])
            expected_tags.append(tag)

            patch_file = "{}/{}".format(self._params.PATCHS_ROOT_PATH, items[0])
            img = io.imread(patch_file, as_grey=False)
            # 样本矩阵化
            # patch = np.array(img)
            net.blobs['data'].data[...] = transformer.preprocess('data', patch)
            net.forward()
            # 样本类别的输出概率值
            prob = net.blobs['prob'].data[0].flatten()
            # 提取难样本

            # 样本识别率大于85%则分类正确
            if prob[0] > 0.85:
               tag = 0
            elif prob[1] > 0.85:
                tag = 1
            else:
                tag = -1

            predicted_tags.append(tag)

        f.close()

        print("Classification report for classifier %s:\n%s\n"
              % (net, metrics.classification_report(expected_tags, predicted_tags)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_tags, predicted_tags))

        return

