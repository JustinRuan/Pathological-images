#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-05-20'

"""

import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunction = CenterlossFunction.apply

    # def forward(self, y, feat):
    #     # To squeeze the Tenosr
    #     batch_size = feat.size(0)
    #     feat = feat.view(batch_size, 1, 1, -1).squeeze()
    #     # To check the dim of centers and features
    #     if feat.size(1) != self.feat_dim:
    #         raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
    #     return self.centerlossfunction(feat, y, self.centers)

    def forward(self, y, feat, weight=None):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(-1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(-1)))
        return self.centerlossfunction(feat, y, self.centers, weight)

class CenterlossFunction(Function):

    # @staticmethod
    # def forward(ctx, feature, label, centers):
    #     ctx.save_for_backward(feature, label, centers)
    #     centers_pred = centers.index_select(0, label.long())
    #     return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0

    @staticmethod
    def forward(ctx, feature, label, centers, weight=None):
        ctx.save_for_backward(feature, label, centers, weight)
        centers_pred = centers.index_select(0, label.long())
        if weight is None:
            return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0
        else:
            return (weight*(feature - centers_pred).pow(2).sum(1)).sum(0) / 2.0


    # @staticmethod
    # def backward(ctx, grad_output):
    #     feature, label, centers, weight = ctx.saved_variables
    #     grad_feature = feature - centers.index_select(0, label.long()) # Eq. 3
    #
    #     # init every iteration
    #     counts = torch.ones(centers.size(0))
    #     grad_centers = torch.zeros(centers.size())
    #     if feature.is_cuda:
    #         counts = counts.cuda()
    #         grad_centers = grad_centers.cuda()
    #     # print counts, grad_centers
    #
    #     # Eq. 4 || need optimization !! To be vectorized, but how?
    #     for i in range(feature.size(0)):
    #         # j = int(label[i].data[0])
    #         j = int(label[i].data.item())
    #         counts[j] += 1
    #         if weight is None:
    #             grad_centers[j] += (centers.data[j] - feature.data[i])
    #         else:
    #             grad_centers[j] += weight.data[i] * (centers.data[j] - feature.data[i])
    #     # print counts
    #     grad_centers = Variable(grad_centers/counts.view(-1, 1))
    #
    #     return grad_feature * grad_output, None, grad_centers, None

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, weight = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long()) # Eq. 3

        # init every iteration
        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()
        # print counts, grad_centers

        # Eq. 4 || need optimization !! To be vectorized, but how?
        # for i in range(feature.size(0)):
        for i in range(label.size(0)):
            # print("feature.size(0)", feature.size(0), "label len ", len(label.data))
            # j = int(label[i].data[0])
            j = int(label.data[i])
            counts[j] += 1
            if weight is None:
                grad_centers[j] += (centers.data[j] - feature.data[i])
            else:
                grad_centers[j] += weight.data[i] * (centers.data[j] - feature.data[i])
            # grad_centers[j] += (centers.data[j] - feature.data[i])

        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers, None