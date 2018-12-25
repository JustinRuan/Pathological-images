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
from torch.nn import functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from .net.ae import Autoencoder, VAE, CAE
from core.util import latest_checkpoint

##################################################################################################################
# Reconstruction + KL divergence losses summed over all elements and batch
def variational_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

################################################################################################################
lam = 1e-4
mse_loss = nn.BCELoss(size_average = False)
# contractive autoencoder, CAE
def contractive_loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss

    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.


    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder

    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term

    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

###################################################################################################################

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
        # convolutional Auto-Encoder
        if self.model_name == "cae":
            model = Autoencoder(self.out_dim)
        # Variational convolutional Auto-Encoder
        elif self.model_name == "vcae":
            model = VAE(self.out_dim)
        # contractive convolutional Auto-Encoder
        elif self.model_name == "ccae":
            model = CAE(self.out_dim)

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

        model = self.load_model()
        print(model)

        if self.use_GPU:
            model.cuda()

        # criterion = nn.BCELoss()
        if self.model_name == "cae":
            criterion = nn.BCELoss(reduction='mean')
        elif self.model_name in ["vcae", "ccae"]:
            criterion = None

        display_criterion = nn.MSELoss() # 附加信息显示用,不计入BP过程

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # ,weight_decay=1e-5
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

            model.train()
            total_loss = 0
            total_loss2 = 0
            for i, (x, _) in enumerate(data_loader):
                if self.use_GPU:
                    b_x = Variable(x).cuda()  # batch x
                else:
                    b_x = Variable(x)  # batch x

                if self.model_name == "cae":
                    out = model(b_x)
                    loss = criterion(out, b_x)
                    loss2 = display_criterion(out, b_x)
                elif self.model_name == "vcae":
                    recon_batch, mu, logvar = model(b_x)
                    loss = variational_loss_function(recon_batch, b_x, mu, logvar)
                    loss2 = display_criterion(recon_batch, b_x)
                elif self.model_name == "ccae":
                    hidden_representation, recons_x = model(b_x)
                    W = model.state_dict()['encoder_fc.fc2.weight']
                    loss = contractive_loss_function(W, b_x.view(-1, self.image_size*self.image_size), recons_x,
                                         hidden_representation, lam)
                    loss2 = display_criterion(recons_x, b_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.model_name == "cae":
                    running_loss = loss.item() * b_x.size(0)
                elif self.model_name == "vcae":
                    running_loss = loss.item()
                elif self.model_name == "ccae":
                    running_loss = loss.item()

                total_loss += running_loss

                running_loss2 = loss2.item() * b_x.size(0)
                total_loss2 += running_loss2
                print('%d / %d ==> Loss: %.4f |  Loss: %.4f'  % (i, iter_per_epoch, running_loss, running_loss2))

                if (i+1) % save_step == 0:
                    if self.model_name == "cae":
                        reconst_images = model(fixed_x)
                    elif self.model_name == "vcae":
                        reconst_images, _, _ = model(fixed_x)
                    elif self.model_name == "ccae":
                        _, reconst_images = model(b_x)

                    torchvision.utils.save_image(reconst_images.data.cpu(),
                                                 pic_path + '/reconst_images_{:04d}_{:06d}.png'.format(epoch + 1, i + 1))

            scheduler.step(total_loss)

            epoch_loss = total_loss / iter_per_epoch
            epoch_loss2 = total_loss2 / iter_per_epoch
            torch.save(model.state_dict(),
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
