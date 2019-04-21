#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-21'

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class ACD_Model(nn.Module):
    def __init__(self, ):
        super(ACD_Model, self).__init__()

        init_varphi = np.asarray([[0.6060, 1.2680, 0.7989],
                                  [1.2383, 1.2540, 0.3927]])
        # alpha = tf.Variable(init_varphi[0], dtype='float32')
        # beta = tf.Variable(init_varphi[1], dtype='float32')
        # w = [tf.Variable(1.0, dtype='float32'), tf.Variable(1.0, dtype='float32'), tf.constant(1.0)]
        alpha = torch.FloatTensor(init_varphi[0])
        beta = torch.FloatTensor(init_varphi[1])
        w = torch.nn.Parameter(torch.FloatTensor([1.0, 1.0]), requires_grad=True)

        # sca_mat = tf.stack((tf.cos(alpha) * tf.sin(beta), tf.cos(alpha) * tf.cos(beta), tf.sin(alpha)), axis=1)
        # cd_mat = tf.matrix_inverse(sca_mat)
        sca_mat = torch.stack([torch.cos(alpha) * torch.sin(beta), torch.cos(alpha) * torch.cos(beta), torch.sin(alpha)],)
        cd_mat = torch.nn.Parameter(torch.inverse(sca_mat), requires_grad=True)

        self.w = w
        self.cd_mat = cd_mat


    def forward(self, x):

        # s = tf.matmul(input_od, cd_mat) * w

        s1 = torch.matmul(x, self.cd_mat)
        one = torch.FloatTensor([1.0])
        one = one.to(x.device)
        s2 = torch.cat((self.w, one))
        out = s1 * s2

        return out

    def loss_function(self, out):
        lambda_p=torch.FloatTensor([0.002])
        lambda_b=torch.FloatTensor([10])
        lambda_e=torch.FloatTensor([1])
        eta=torch.FloatTensor([0.6])
        gamma=torch.FloatTensor([0.5])

        lambda_p = lambda_p.to(out.device)
        lambda_b = lambda_b.to(out.device)
        lambda_e = lambda_e.to(out.device)
        eta = eta.to(out.device)
        gamma = gamma.to(out.device)


        # h, e, b = tf.split(s, (1, 1, 1), axis=1)
        h, e, b = torch.split(out, 1, dim=1)

        # l_p1 = tf.reduce_mean(tf.square(b))
        # l_p2 = tf.reduce_mean(2 * h * e / (tf.square(h) + tf.square(e)))
        # l_b = tf.square((1 - eta) * tf.reduce_mean(h) - eta * tf.reduce_mean(e))
        # l_e = tf.square(gamma - tf.reduce_mean(s))
        l_p1 = torch.mean(torch.mul(b, b))
        l_p2 = torch.mean(2 * h * e / (torch.mul(h, h) + torch.mul(e, e)))
        l_bb = (1 - eta) * torch.mean(h) - eta * torch.mean(e)
        l_b = torch.mul(l_bb, l_bb)
        l_ee = gamma - torch.mean(out)
        l_e = torch.mul(l_ee, l_ee)

        # objective = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e
        # target = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(objective)
        #
        # return target, cd_mat, w

        loss = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e

        return loss

    # def get_params(self):
    #     w = self.w.data[0].
    #     return
