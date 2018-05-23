#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-05-20'

"""

import numpy as np
from PIL import Image
import io


class Block(object):
    # x is col, y is row
    def __init__(self, snumber, x, y, scale, opcode, w, h):
        self.slice_number = snumber
        self.x = x
        self.y = y
        self.scale = scale
        self.opcode = opcode
        self.width = w
        self.height = h
        self.img_file = None
        self.img = None

    def updateXY(self, x ,y):
        self.x = x
        self.y = y

    def encoding(self):
        intScale = np.rint(self.scale * 100).astype(np.int)
        return "{0}_{1:0>6}_{2:0>6}_{3:0>4}_{4:0>1}".format(self.slice_number,
                                                            self.x, self.y, intScale, self.opcode)

    def set_img_file(self, image_file):
        self.img_file = image_file

    def set_img(self, img):
        self.img = img

    def get_img(self):
        if self.img == None:
            self.img = Image.open(io.BytesIO(self.img_file))

        return self.img

    def save_img(self, path):
        filename = '/{}.jpg'.format(self.encoding())
        self.img_file.tofile(path + filename)
