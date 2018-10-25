#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import numpy as np
from scipy.ndimage.filters import sobel, gaussian_filter

class FastRadialSymmetryTransform(object):
    def __init__(self, gradient_function=None):
        self._index_cache = {}
        if gradient_function is None:
            gradient_function = self.sobel_gradient
        self.gradient_function = gradient_function

    def index_arrays(self, shape):
        if shape not in self._index_cache:
            self._index_cache[shape] = np.meshgrid(np.arange(shape[1]),
                                                   np.arange(shape[0]))
        return self._index_cache[shape]

    # def pixel_map(self, gmag, gx, gy, r):
    #     x, y = self.index_arrays(gmag.shape)
    #     gx = gx / (gmag + 1e-5)
    #     gy = gy / (gmag + 1e-5)
    #
    #     nx = np.rint(gx * r + 0.5).astype(np.int64)
    #     ny = np.rint(gy * r + 0.5).astype(np.int64)
    #     posx = x + nx
    #     posy = y + ny
    #     negx = x - nx
    #     negy = y - ny
    #     return posx, posy, negx, negy

    def pixel_map(self, gmag, gx, gy, r):
        x, y = self.index_arrays(gmag.shape)

        gx = gx / (gmag + 1e-5)
        gy = gy / (gmag + 1e-5)

        posxLst = []
        posyLst = []
        negxLst = []
        negyLst = []
        for n in range(1, r + 1):
            nx = np.rint(gx * n + 0.5).astype(np.int64)
            ny = np.rint(gy * n + 0.5).astype(np.int64)
            posxLst.append(x + nx)
            posyLst.append(y + ny)
            negxLst.append(x - nx)
            negyLst.append(y - ny)

        return posxLst, posyLst, negxLst, negyLst

    # def transform_component(self, mag, gx, gy, n, sigma, alpha):
    #     posx, posy, negx, negy = self.pixel_map(mag, gx, gy, n)
    #     orientation, magnitude = self.orientation_and_magnitude(
    #         posx, posy, negx, negy, mag, n)
    #     o_max = np.max(np.abs(orientation))
    #     m_max = np.max(np.abs(magnitude))
    #
    #     # orientation = np.abs(orientation) / o_max
    #     orientation = 255 * np.abs(orientation) / o_max
    #     #
    #     magnitude = np.abs(magnitude) / m_max
    #     magnitude[magnitude < 0.05] = 0
    #     magnitude = 255 * magnitude
    #
    #     F = self.compute_F(orientation, magnitude, alpha)
    #
    #     return gaussian_filter(F,sigma = sigma, truncate = 4.0)

    def transform_component(self, image, mag, gx, gy, n, sigma, alpha):
        posx, posy, negx, negy = self.pixel_map(mag, gx, gy, n)
        orientation, magnitude = self.orientation_and_magnitude(
            posx, posy, negx, negy, mag, n)

        o_max = np.max(orientation)
        m_max = np.max(magnitude)
        o_min = 0    #np.min(orientation)
        m_min = 0    #np.min(magnitude)
        orientation = 1 * (orientation - o_min) / (o_max - o_min)
        magnitude = 1 * (magnitude - m_min) / (m_max - m_min)

        F = self.compute_F(image, orientation, magnitude, alpha)

        return gaussian_filter(F,sigma = sigma, truncate = 4.0)

    def transform(self, image, ns, sigmas, alpha):
        transform = np.zeros(image.shape)
        mag, gx, gy = self.gradient_function(image)
        sigmas = sigmas * ns
        insk = zip(range(len(ns)), ns, sigmas)
        for i, n, s, in insk:
            transform += self.transform_component(
                image, mag, gx, gy, n, s, alpha)
        return transform

    def sobel_gradient(self, image):
        grad_y = sobel(image, 0)
        grad_x = sobel(image, 1)
        mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
        return mag, grad_x, grad_y


    # def orientation_and_magnitude(self, posx, posy, negx, negy, mag, r):
    #     orientation = np.zeros(mag.shape)
    #     magnitude = np.zeros(mag.shape)
    #     h, w = mag.shape
    #
    #     posx[posx < 0] = 0
    #     posy[posy < 0] = 0
    #     negx[negx < 0] = 0
    #     negy[negy < 0] = 0
    #     posx[posx > w - 1] = w - 1
    #     posy[posy > h - 1] = h - 1
    #     negx[negx > w - 1] = w - 1
    #     negy[negy > h - 1] = h - 1
    #     np.add.at(orientation, [posy, posx], 1)
    #     # np.add.at(orientation, [negy, negx], -1)//对应黑色区域
    #     np.add.at(magnitude, [posy, posx], mag)
    #     # np.add.at(magnitude, [negy, negx], -mag)//对应黑色区域
    #     orientation[:,0] = 0
    #     orientation[:, w - 1] = 0
    #     orientation[0, :] = 0
    #     orientation[h - 1, :] = 0
    #     magnitude[:,0] = 0
    #     magnitude[:, w - 1] = 0
    #     magnitude[0, :] = 0
    #     magnitude[h - 1, :] = 0
    #     return orientation, magnitude

    def orientation_and_magnitude(self, posxLst, posyLst, negxLst, negyLst, mag, r):
        orientation = np.zeros(mag.shape)
        magnitude = np.zeros(mag.shape)
        h, w = mag.shape

        for n in range(r) :
            posx = posxLst[n]
            posy = posyLst[n]
            negx = negxLst[n]
            negy = negyLst[n]
            posx[posx < 0] = 0
            posy[posy < 0] = 0
            negx[negx < 0] = 0
            negy[negy < 0] = 0
            posx[posx > w - 1] = w - 1
            posy[posy > h - 1] = h - 1
            negx[negx > w - 1] = w - 1
            negy[negy > h - 1] = h - 1
            np.add.at(orientation, [posy, posx], 1)
            # np.add.at(orientation, [negy, negx], -1)#对应黑色区域
            np.add.at(magnitude, [posy, posx], mag)
            # np.add.at(magnitude, [negy, negx], -mag)#对应黑色区域


        orientation[:,0] = 0
        orientation[:, w - 1] = 0
        orientation[0, :] = 0
        orientation[h - 1, :] = 0
        magnitude[:,0] = 0
        magnitude[:, w - 1] = 0
        magnitude[0, :] = 0
        magnitude[h - 1, :] = 0
        return orientation, magnitude

    def compute_F(self, image, orientation, magnitude, alpha):
        s = (orientation ** alpha) * (magnitude + image)
        return s
