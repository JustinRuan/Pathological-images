#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2019-04-16'

"""

'''
Jeroen A.W.M. van der Laak, Pahlplatz M M M , Hanselaar A G J M , et al. 
Hue-saturation-density (HSD) model for stain recognition in digital images from transmitted light microscopy[J]. 
Cytometry, 2000, 39(4):275-284.
'''
import numpy as np
import math

def rgb2hsd(rgb):
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    DB = - (np.log((B + 1.0) / 257.0))
    DG = - (np.log((G + 1.0) / 257.0))
    DR = - (np.log((R + 1.0) / 257.0))

    D = (DR + DB + DG) / 3.0
    Cx = DR / D - 1
    Cy = (DG - DB) / (D * math.sqrt(3))

    # hsd = np.zeros(np.shape(rgb))
    # hsd[:, :, 0] = Cx
    # hsd[:, :, 1] = Cy
    # hsd[:, :, 2] = D

    hsd = np.dstack([Cx, Cy, D])
    return hsd


def hsd2rgb(hsd):
# hsd color space to rgb color space
#input:(Cx, Cy, D) in the hsd color space
#output:rgb

    Cx = hsd[:, :, 0]
    Cy = hsd[:, :, 1]
    D = hsd[:, :, 2]
    Dr = (Cx + 1) * D
    Dg = ((Cy * D * math.sqrt(3)) + (3 * D - Dr)) / 2
    Db = 3 * D - Dr - Dg

    b = np.exp(-Db) * 257.0 - 1
    g = np.exp(-Dg) * 257.0 - 1
    r = np.exp(-Dr) * 257.0 - 1

    r = np.clip(r / 255, 0, 1).astype(np.float)
    g = np.clip(g / 255, 0, 1).astype(np.float)
    b = np.clip(b / 255, 0, 1).astype(np.float)

    # rgb = np.zeros(np.shape(hsd), dtype=np.int)
    # rgb[:, :, 0] = r
    # rgb[:, :, 1] = g
    # rgb[:, :, 2] = b
    rgb = np.dstack([r, g, b])

    return rgb


def RGB2HSD2(X):
    XX = X / 255.0
    eps = np.finfo(float).eps
    XX[np.where(XX == 0.0)] = eps

    OD = -np.log(XX / 1.0)
    D = np.mean(OD, axis=2)
    D[np.where(D == 0.0)] = eps

    cx = OD[:, :, 0] / (D) - 1.0
    cy = (OD[ :, :, 1] - OD[ :, :, 2]) / (np.sqrt(3.0) * D)

    D = np.expand_dims(D, 2)
    cx = np.expand_dims(cx, 2)
    cy = np.expand_dims(cy, 2)

    X_HSD = np.concatenate((cx, cy, D), 2)
    return X_HSD

def HSD2RGB2(X_HSD):
    X_HSD_0 = X_HSD[..., 2]
    X_HSD_1 = X_HSD[..., 0]
    X_HSD_2 = X_HSD[..., 1]
    D_R = np.expand_dims(np.multiply(X_HSD_1 + 1, X_HSD_0), 2)
    D_G = np.expand_dims(np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 + np.sqrt(3.0) * X_HSD_2), 2)
    D_B = np.expand_dims(np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 - np.sqrt(3.0) * X_HSD_2), 2)

    X_OD = np.concatenate((D_R, D_G, D_B), axis=2)
    X_RGB = 1.0 * np.exp(-X_OD)

    return X_RGB

if __name__ == '__main__':
    from skimage import data, io

    image = data.astronaut() # RGB image
    hsd_img = rgb2hsd(image)
    rgb_img = hsd2rgb(hsd_img)
    hsd_img2 = RGB2HSD2(image)
    rgb_img2 = HSD2RGB2(hsd_img2)
    io.imshow_collection([image, hsd_img, rgb_img, hsd_img2, rgb_img2])
    io.show()

