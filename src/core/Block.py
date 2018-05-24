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
    def __init__(self, snumber="", x=0, y=0, scale=0, opcode=0, w=0, h=0):
        self.slice_number = snumber
        self.x = x
        self.y = y
        self.scale = scale
        self.opcode = opcode
        self.width = w
        self.height = h
        self.img_file = None
        self.img = None

    def decoding(self, filename, w, h):
        left = filename.rfind("/")
        right = filename.index(".")
        file_code = filename[left + 1 : right]
        code = file_code.split("_")

        self.slice_number = code[0]
        self.x = int(code[1])
        self.y = int(code[2])
        self.scale = float(code[3])/100
        self.opcode = int(code[4])
        self.w = w
        self.h = h

        return

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
            self.img = np.array(Image.open(io.BytesIO(self.img_file)))

        return self.img

    def save_img(self, path):
        filename = '/{}.jpg'.format(self.encoding())
        self.img_file.tofile(path + filename)
        return

    def load_img(self, filename):
        self.img = Image.open(filename)
        (w, h) = self.img.size
        self.decoding(filename, w, h)
        return