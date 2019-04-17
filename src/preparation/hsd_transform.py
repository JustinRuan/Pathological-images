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

if __name__ == '__main__':
    from skimage import data, io

    image = data.astronaut() # RGB image
    hsd_img = rgb2hsd(image)
    rgb_img = hsd2rgb(hsd_img)
    # io.imshow(rgb_img)
    io.imshow_collection([image, hsd_img, rgb_img, rgb_img])
    io.show()