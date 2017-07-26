#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:HeSimin
import numpy as np
from scipy.ndimage.filters import sobel, gaussian_filter
import cv2

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

    def pixel_map(self, gmag, gx, gy, n):
        x, y = self.index_arrays(gmag.shape)
        gx = gx / gmag
        gy = gy / gmag
        nx = np.rint(gx * n).astype(np.int64)
        ny = np.rint(gy * n).astype(np.int64)
        posx = x + nx
        posy = y + ny
        negx = x - nx
        negy = y - ny
        return posx, posy, negx, negy

    def transform(self, inputImage, alpha, n):
        mag, gx, gy = self.gradient_function(inputImage)
        posx, posy, negx, negy = self.pixel_map(mag, gx, gy, n)
        orientation, magnitude = self.orientation_and_magnitude(
            posx, posy, negx, negy, mag)
        o_max = np.max(np.abs(orientation))
        m_max = np.max(np.abs(magnitude))
        orientation = np.abs(orientation) / o_max
        magnitude = np.abs(magnitude) / m_max
        F  = (orientation ** alpha)* magnitude
        outputImage = cv2.GaussianBlur(F, (7, 7), 5)
        return outputImage

    def sobel_gradient(self, image):
        grad_y = sobel(image, 0)
        grad_x = sobel(image, 1)
        mag = np.sqrt(grad_y ** 2 + grad_x ** 2) + 1e-16
        return mag, grad_x, grad_y

    def orientation_and_magnitude(self, posx, posy, negx, negy, mag):
        orientation = np.zeros(mag.shape)
        magnitude = np.zeros(mag.shape)
        h, w = mag.shape
        posx[posx < 0] = 0
        posy[posy < 0] = 0
        negx[negx < 0] = 0
        negy[negy < 0] = 0
        posx[posx > w - 1] = w - 1
        posy[posy > h - 1] = h - 1
        negx[negx > w - 1] = w - 1
        negy[negy > h - 1] = h - 1
        np.add.at(orientation, [posy, posx], 1)
        np.add.at(orientation, [negy, negx], -1)
        np.add.at(magnitude, [posy, posx], mag)
        np.add.at(magnitude, [negy, negx], -mag)
        return orientation, magnitude