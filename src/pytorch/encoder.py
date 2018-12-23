#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-12-17'

"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from .net.ae import Autoencoder
from core.util import latest_checkpoint


class Encoder(object):
    def __init__(self, params, model_name, patch_type):
        '''
        初始化
        :param params: 系统参数
        :param model_name: 调用的编码器算法代号
        :param patch_type: 训练编码器所用数据集的代号
        '''
        self._params = params
        self.model_name = model_name
        self.patch_type = patch_type
        self.NUM_WORKERS = params.NUM_WORKERS

        if self.patch_type == "cifar10":
            self.out_dim = 32
            self.image_size = 32

        self.model_root = "{}/models/pytorch/{}_{}".format(self._params.PROJECT_ROOT, self.model_name, self.patch_type)

        self.use_GPU = True

    def create_initial_model(self):
        '''
        构造编码器
        :return:
        '''
        if self.model_name == "cae":
            model = Autoencoder(self.out_dim)
        # elif self.model_name == "cae2":
            # model = Autoencoder2(self.out_dim)
        return model

    def load_model(self, model_file = None):
        '''
        加载模型
        :param model_file: 指定模型的存盘文件
        :return:
        '''
        model = self.create_initial_model()

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            model.load_state_dict(torch.load(model_file))
        else:
            checkpoint_dir = self.model_root
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                model.load_state_dict(torch.load(latest))

        return model

    def train_ae(self, batch_size=100, epochs=20):
        '''
        训练编码器
        :param batch_size: 每批的图片数量
        :param epochs: epoch数
        :return:
        '''
        data_root = os.path.join(os.path.expanduser('~'), '.keras/datasets/')  # 共用Keras下载的数据

        transform = transforms.Compose([
                # transforms.Resize((8, 8), interpolation=0),
                # transforms.Resize((32, 32), interpolation=0),
                transforms.ToTensor(),
            ])

        if self.patch_type == "cifar10":
            train_data = torchvision.datasets.cifar.CIFAR10(
                root=data_root,  # 保存或者提取位置
                train=True,  # this is training data
                transform=transform,
                download=False
            )

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        ae = self.load_model()
        print(ae)

        if self.use_GPU:
            ae.cuda()

        criterion = nn.BCELoss()
        display_criterion = nn.MSELoss() # 附加信息显示用,不计入BP过程

        optimizer = torch.optim.Adam(ae.parameters(), lr=0.001) # ,weight_decay=1e-5
        # optimizer = torch.optim.Adadelta(ae.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose = True,
                                                               factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
        iter_per_epoch = len(data_loader)

        data_iter = iter(data_loader)
        # save fixed inputs for debugging
        fixed_x, _ = next(data_iter)

        pic_path = self.model_root + '/data'
        if (not os.path.exists(pic_path)):
            os.makedirs(pic_path)
        torchvision.utils.save_image(Variable(fixed_x).data.cpu(), pic_path + '/real_images.png')

        save_step = iter_per_epoch - 1

        if self.use_GPU:
            fixed_x = Variable(fixed_x).cuda()
        else:
            fixed_x = Variable(fixed_x)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            ae.train()
            total_loss = 0
            total_loss2 = 0
            for i, (x, _) in enumerate(data_loader):
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                else:
                    b_x = Variable(x)  # batch x

                out = ae(b_x)
                loss = criterion(out, b_x)
                loss2 = display_criterion(out, b_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss = loss.item() * b_x.size(0)
                total_loss += running_loss

                running_loss2 = loss2.item() * b_x.size(0)
                total_loss2 += running_loss2
                print('%d / %d ==> Loss: %.4f |  Loss: %.4f'  % (i, iter_per_epoch, running_loss, running_loss2))

                if (i+1) % save_step == 0:
                    reconst_images = ae(fixed_x)
                    torchvision.utils.save_image(reconst_images.data.cpu(),
                                                 pic_path + '/reconst_images_{:04d}_{:06d}.png'.format(epoch + 1, i + 1))

            scheduler.step(total_loss)

            epoch_loss = total_loss / iter_per_epoch
            epoch_loss2 = total_loss2 / iter_per_epoch
            torch.save(ae.state_dict(),
                           self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss, epoch_loss2))

    def extract_feature(self, image_itor, seeds_num, batch_size):
        '''
        使用编码器，提取图块的特征向量
        :param image_itor: 图块迭代器
        :param seeds_num: 图块的总数
        :param batch_size: 每批的图块数量
        :return: 特征向量集
        '''
        ae = self.load_model()
        for param in ae.parameters():
            param.requires_grad = False

        print(ae)

        if self.use_GPU:
            ae.cuda()
        ae.eval()

        data_len = seeds_num // batch_size + 1
        results = []

        for step, x in enumerate(image_itor):
            if self.use_GPU:
                b_x = Variable(x).cuda()  # batch x
            else:
                b_x = Variable(x)  # batch x

            output = ae.encode(b_x)
            results.extend(output.cpu().numpy())
            print('encoding => %d / %d ' % (step + 1, data_len))

        return  results
