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
from pytorch.net import Autoencoder, VAE, CAE
from core.util import latest_checkpoint, read_csv_file, clean_checkpoint
from torchsummary import summary
from pytorch.image_dataset import Image_Dataset
from skimage.io import imread
from pytorch.net import Encoder, Decoder, Discriminator

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
mse_loss = nn.BCELoss(reduction = "sum")
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
RHO = 0.01
# Sparse Autoencoder
def sparse_loss_function(z):
    p = torch.FloatTensor([RHO for _ in range(len(z[0]))]).unsqueeze(0)
    p = p.to(z.device)
    q = torch.sum(z, dim=0, keepdim=True)

    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    sparsity_penalty = s1 + s2

    return sparsity_penalty

###################################################################################################################

class EncoderFactory(object):
    def __init__(self, params, model_name, patch_type, out_dim = 32):
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
            self.latent_vector_dim = out_dim
            self.image_size = 32
        elif self.patch_type == "AE_500_32":
            self.latent_vector_dim = out_dim
            self.image_size = 32

        self.model_root = "{}/models/pytorch/{}_{}_{}".format(self._params.PROJECT_ROOT, self.model_name,
                                                              self.patch_type, self.latent_vector_dim)

        self.use_GPU = True
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if self.use_GPU else "cpu")

    def create_initial_model(self):
        '''
        构造编码器
        :return:
        '''
        # convolutional Auto-Encoder
        if self.model_name == "cae":
            model = Autoencoder(self.latent_vector_dim)
        # Variational convolutional Auto-Encoder
        elif self.model_name == "vcae":
            model = VAE(self.latent_vector_dim)
        # contractive convolutional Auto-Encoder
        elif self.model_name == "ccae":
            model = CAE(self.latent_vector_dim)
        # Sparse convolutional Autoencoder
        elif self.model_name == "scae":
            model = CAE(self.latent_vector_dim)

        return model

    def load_model(self, model_file = None):
        '''
        加载模型
        :param model_file: 指定模型的存盘文件
        :return:
        '''
        if self.model_name == "aae":
            model, _, _ = self.load_adversarial_model(model_file)
        else:
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

    def load_train_data(self):
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
        elif self.patch_type == "AE_500_32":
            # data_list = "{}/{}.txt".format(self._params.PATCHS_ROOT_PATH, "AE_500_32")
            # Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, data_list)
            # train_data = Image_Dataset(Xtrain, Ytrain)
            train_data = self.load_custom_data_to_memory(self.patch_type)

        return train_data

    def train_ae(self, batch_size=64, epochs=20):
        '''
        训练编码器
        :param batch_size: 每批的图片数量
        :param epochs: epoch数
        :return:
        '''
        train_data = self.load_train_data()

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        model = self.load_model()
        summary(model, input_size=(3, self.image_size, self.image_size), device="cpu")

        # if self.use_GPU:
        #     model.cuda()
        model.to(self.device)

        # criterion = nn.BCELoss()
        if self.model_name in ["cae"]:
            criterion = nn.BCELoss(reduction='mean')
        elif self.model_name in ["vcae", "ccae"]:
            criterion = None
        elif self.model_name in ["scae"]:
            criterion = nn.MSELoss(reduction='mean')

        display_criterion = nn.MSELoss() # 附加信息显示用,不计入BP过程

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # ,weight_decay=1e-5
        # optimizer = torch.optim.Adadelta(ae.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose = True,
                                                               factor=0.1)  # mode为min，则loss不下降学习率乘以factor，max则反之
        iter_per_epoch = len(data_loader)
        data_size = len(train_data)

        data_iter = iter(data_loader)
        # save fixed inputs for debugging
        fixed_x, _ = next(data_iter)

        pic_path = self.model_root + '/data'
        if (not os.path.exists(pic_path)):
            os.makedirs(pic_path)
        torchvision.utils.save_image(Variable(fixed_x).data.cpu(), pic_path + '/real_images.png')

        save_step = iter_per_epoch - 1

        # if self.use_GPU:
        #     fixed_x = Variable(fixed_x).cuda()
        # else:
        #     fixed_x = Variable(fixed_x)
        fixed_x = Variable(fixed_x.to(self.device))

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            model.train()
            total_loss = 0
            total_loss2 = 0
            for i, (x, _) in enumerate(data_loader):
                # if self.use_GPU:
                #     b_x = Variable(x).cuda()  # batch x
                # else:
                #     b_x = Variable(x)  # batch x
                b_x = Variable(x.to(self.device))

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
                elif self.model_name == "scae":
                    hidden_representation, recons_x = model(b_x)
                    loss = criterion(recons_x, b_x) + 3 * sparse_loss_function(hidden_representation)
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
                elif self.model_name == "scae":
                    running_loss = loss.item() * b_x.size(0)

                total_loss += running_loss

                running_loss2 = loss2.item() * b_x.size(0)
                total_loss2 += running_loss2
                print('%d / %d ==> Loss: %.4f | MSE Loss: %.4f'  % (i, iter_per_epoch, running_loss, running_loss2))

                if (i+1) % save_step == 0:
                    if self.model_name == "cae":
                        reconst_images = model(fixed_x)
                    elif self.model_name == "vcae":
                        reconst_images, _, _ = model(fixed_x)
                    elif self.model_name == "ccae":
                        _, reconst_images = model(fixed_x)
                    elif self.model_name == "scae":
                        _, reconst_images = model(fixed_x)

                    torchvision.utils.save_image(reconst_images.data.cpu(),
                                                 pic_path + '/reconst_images_{:04d}_{:06d}.png'.format(epoch + 1, i + 1))

            scheduler.step(total_loss)

            epoch_loss = total_loss / data_size
            epoch_loss2 = total_loss2 / data_size
            torch.save(model.state_dict(),
                           self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch + 1, epoch_loss, epoch_loss2))

    def load_custom_data_to_memory(self, samples_name):
        data_list = "{}/{}.txt".format(self._params.PATCHS_ROOT_PATH, samples_name)
        Xtrain, Ytrain = read_csv_file(self._params.PATCHS_ROOT_PATH, data_list)
        # train_data = Image_Dataset(Xtrain, Ytrain)
        img_data = []
        for file_name in Xtrain:
            img = imread(file_name) / 255
            img_data.append(img)

        img_numpy = np.array(img_data).transpose((0, 3, 1, 2))
        label_numpy = np.array(Ytrain)
        train_data = torch.utils.data.TensorDataset(torch.from_numpy(img_numpy).float(),
                                                    torch.from_numpy(label_numpy).long())
        return train_data

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

        summary(ae, input_size=(3, self.image_size, self.image_size), device="cpu")

        # if self.use_GPU:
        #     ae.cuda()
        ae.to(self.device)
        ae.eval()

        data_len = seeds_num // batch_size + 1
        results = []

        for step, x in enumerate(image_itor):
            # if self.use_GPU:
            #     b_x = Variable(x).cuda()  # batch x
            # else:
            #     b_x = Variable(x)  # batch x
            b_x = Variable(x.to(self.device))

            output = ae.encode(b_x)
            results.extend(output.cpu().numpy())
            print('encoding => %d / %d ' % (step + 1, data_len))

        return  results

    def eval_latent_vector_loss(self, batch_size):
        train_data = self.load_train_data()

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,  drop_last=True)
        data_size = len(data_loader) * batch_size

        model = self.load_model()
        model.to(self.device)
        model.eval()

        criterion = nn.MSELoss(reduction='mean')
        results = []

        for dim_index in range(self.latent_vector_dim):
            # index = torch.tensor([dim_index]).to(self.device)
            index_numpy = np.full((batch_size, 1), dim_index)
            index = torch.from_numpy(index_numpy).long()
            index = index.to(self.device)

            total_loss = 0
            for i, (x, _) in enumerate(data_loader):
                b_x = Variable(x.to(self.device))

                if self.model_name == "cae":
                    h = model.encode(b_x)

                    # h.index_fill_(1, index, 0)
                    rnd = torch.randn(batch_size).unsqueeze(1).to(self.device)
                    h.scatter_(1, index, rnd)

                    out = model.decode(h)
                    loss = criterion(out, b_x)

                if self.model_name == "cae":
                    running_loss = loss.item() * b_x.size(0)

                total_loss += running_loss

            avg_loss =  total_loss / data_size
            results.append(avg_loss)
            print('suppressing dim %d ==> MSE Loss: %.6f' % (dim_index, avg_loss))

            # max_loss = np.max(results)
            # min_loss = np.min(results)
        return results

##################################################################################################################
#######################  Adversarial Autoencoder #################################################################
##################################################################################################################
    def load_adversarial_model(self, model_file=None):
        Q_encoder = Encoder(self.latent_vector_dim)
        P_decoder = Decoder(self.latent_vector_dim)
        D_guess = Discriminator(self.latent_vector_dim)

        if model_file is not None:
            print("loading >>> ", model_file, " ...")
            save_model = torch.load(model_file)
            Q_encoder.load_state_dict(save_model["encoder"])
            P_decoder.load_state_dict(save_model["decoder"])
            D_guess.load_state_dict(save_model["guess"])
        else:
            checkpoint_dir = self.model_root
            if (not os.path.exists(checkpoint_dir)):
                os.makedirs(checkpoint_dir)

            latest = latest_checkpoint(checkpoint_dir)
            if latest is not None:
                print("loading >>> ", latest, " ...")
                save_model = torch.load(latest)
                Q_encoder.load_state_dict(save_model["encoder"])
                P_decoder.load_state_dict(save_model["decoder"])
                D_guess.load_state_dict(save_model["guess"])

        return Q_encoder, P_decoder, D_guess

    def save_adversarial_model(self, encoder, decoder, guess, epoch, loss1, loss2):
        save_filename = self.model_root + "/cp-{:04d}-{:.4f}-{:.4f}.pth".format(epoch + 1, loss1, loss2)
        save_object = {"encoder":encoder.state_dict(),
                       "decoder":decoder.state_dict(),
                       "guess":guess.state_dict()}
        torch.save(save_object, save_filename)


    def train_adversarial_ae(self, batch_size=64, epochs=20):

        train_data = self.load_train_data()

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        Q_encoder, P_decoder, D_guess = self.load_adversarial_model()
        summary(Q_encoder, input_size=(3, self.image_size, self.image_size), device="cpu")
        summary(P_decoder, input_size=(self.latent_vector_dim,), device="cpu")
        summary(D_guess, input_size=(self.latent_vector_dim,), device="cpu")

        Q_encoder.to(self.device)
        P_decoder.to(self.device)
        D_guess.to(self.device)

        Q_solver = torch.optim.Adam(Q_encoder.parameters(), lr=1e-3)
        P_solver = torch.optim.Adam(P_decoder.parameters(), lr=1e-3)
        D_solver = torch.optim.Adam(D_guess.parameters(), lr=1e-4)

        iter_per_epoch = len(data_loader)
        data_size = len(train_data)

        data_iter = iter(data_loader)
        # save fixed inputs for debugging
        fixed_x, _ = next(data_iter)

        pic_path = self.model_root + '/data'
        if (not os.path.exists(pic_path)):
            os.makedirs(pic_path)
        torchvision.utils.save_image(Variable(fixed_x).data.cpu(), pic_path + '/real_images.png')

        save_step = iter_per_epoch - 1
        fixed_x = Variable(fixed_x.to(self.device))

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 80)

            total_loss = 0
            # total_loss2 = 0
            for i, (x, _) in enumerate(data_loader):
                """ Reconstruction phase """
                Q_encoder.train()
                P_decoder.train()

                b_x = Variable(x.to(self.device))

                z_sample = Q_encoder(b_x)

                X_sample = P_decoder(z_sample)
                recon_loss = F.mse_loss(X_sample, b_x, reduction='mean')

                recon_loss.backward()
                P_solver.step()
                Q_solver.step()
                Q_encoder.zero_grad()
                P_decoder.zero_grad()

                # save
                running_loss = recon_loss.item() * b_x.size(0)
                total_loss += running_loss
                print('%d / %d ==> Loss: %.4f | MSE  Loss: %.4f' % (i, iter_per_epoch, running_loss, running_loss))

                """ Regularization phase """
                # Discriminator
                Q_encoder.eval()
                D_guess.train()
                for _ in range(5):
                    z_real = Variable(torch.randn(batch_size, self.latent_vector_dim))
                    z_real = z_real.to(self.device)

                    z_fake = Q_encoder(b_x)

                    D_real = D_guess(z_real)
                    D_fake = D_guess(z_fake)

                    # D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
                    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

                    D_loss.backward()
                    D_solver.step()

                    # Weight clipping
                    for p in D_guess.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    Q_encoder.zero_grad()
                    D_guess.zero_grad()

                """ Generator """
                Q_encoder.train()
                D_guess.eval()

                z_fake = Q_encoder(b_x)
                D_fake = D_guess(z_fake)

                # G_loss = -torch.mean(torch.log(D_fake))
                G_loss = -torch.mean(D_fake)

                G_loss.backward()
                Q_solver.step()
                Q_encoder.zero_grad()
                D_guess.zero_grad()

                # output image
                if (i+1) % save_step == 0:
                    reconst_images = P_decoder(Q_encoder(fixed_x))
                    torchvision.utils.save_image(reconst_images.data.cpu(),
                                                 pic_path + '/reconst_images_{:04d}_{:06d}.png'.format(epoch + 1, i + 1))

            # save model
            epoch_loss = total_loss / data_size
            # epoch_loss2 = total_loss2 / data_size

            self.save_adversarial_model(Q_encoder, P_decoder, D_guess, epoch, epoch_loss, epoch_loss)

        clean_checkpoint(self.model_root, best_number=10)
