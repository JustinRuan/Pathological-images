#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Justin

import numpy as np
import ctypes
from ctypes.wintypes import LPARAM
from ctypes import c_char_p, c_int, c_float, c_double, c_bool, c_ubyte
from ctypes import POINTER, byref
from PIL import Image
import io
import xml.dom.minidom
from skimage import draw
import utils

TUMOR_RANGE_COLOR = 4278190335
NORMAL_RANGE_COLOR = 4278222848

# 定义结构体
class ImageInfoStruct(ctypes.Structure):
    _fields_ = [("DataFilePTR", LPARAM)]


class DigitalSlide(object):
    def __init__(self):
        self._Objdll_ = ctypes.windll.LoadLibrary(utils.KFB_SDK_PATH + "/x64/ImageOperationLib")

        '''
        bool InitImageFileFunc( ImageInfoStruct& sImageInfo, constchar* Path );
        参数：
        1.sImageInfo：返回数字图像文件指针
        2.Path ：数字图像路径
        '''
        # 定义返回类型和参数类型
        self.InitImageFileFunc = self._Objdll_.InitImageFileFunc
        self.InitImageFileFunc.restype = c_bool
        self.InitImageFileFunc.argtypes = [POINTER(ImageInfoStruct), c_char_p]

        '''
        bool GetHeaderInfoFunc( ImageInfoStruct sImageInfo, int&khiImageHeight,
              int	&khiImageWidth,int	&khiScanScale,float	&khiSpendTime,
              double	&khiScanTime,float	&khiImageCapRes,int&khiImageBlockSize);

        1.sImageInfo：传入图像数据指针
        2.khiImageHeight：返回扫描高度
        3.khiImageWidth：返回扫描宽度
        4.khiScanScale：返回扫描倍率
        5.khiSpendTime：返回扫描时间
        6.khiScanTime:返回扫描时间
        7.khiImageCapRes:返回扫描像素与um的比例
        8.khiImageBlockSize:返回扫描块大小
        '''
        self.GetHeaderInfoFunc = self._Objdll_.GetHeaderInfoFunc
        # print("GetHeaderInfoFunc ", self._Objdll_.GetHeaderInfoFunc)
        self.GetHeaderInfoFunc.restype = ctypes.c_bool
        self.GetHeaderInfoFunc.argtypes = [ImageInfoStruct, POINTER(c_int), POINTER(c_int), \
                                           POINTER(c_int), POINTER(c_float), POINTER(c_double), \
                                           POINTER(c_float), POINTER(c_int)]

        '''
        bool UnInitImageFileFunc( ImageInfoStruct& sImageInfo );

               参数：
               1.sImageInfo ： 传入数字图像文件指针
        '''
        self.UnInitImageFileFunc = self._Objdll_.UnInitImageFileFunc
        self.UnInitImageFileFunc.restype = c_bool
        self.UnInitImageFileFunc.argtypes = [POINTER(ImageInfoStruct)]

        '''
        bool GetImageDataRoiFunc( ImageInfoStruct sImageInfo, float fScale,
        int sp_x, int sp_y, int nWidth, int nHeight,
        BYTE** pBuffer, int&DataLength, bool flag);

        参数：
        1.	sImageInfo：传入图像数据指针
        2.	fScale：传入倍率
        3.	sp_x：左上角X坐标
        4.	sp_y：右上角Y坐标
        5.	nWidth：宽度
        6.	nHeight：高度
        7.	pBuffer：返回图像数据指针
        8.	DataLength：返回图像字节长度
        9.	flag：true

        '''
        self.GetImageDataRoiFunc = self._Objdll_.GetImageDataRoiFunc
        # print("GetImageDataRoiFunc ", self._Objdll_.GetImageDataRoiFunc)
        self.GetImageDataRoiFunc.restype = c_bool
        self.GetImageDataRoiFunc.argtypes = [ImageInfoStruct, c_float, c_int, c_int, c_int, c_int, \
                                             POINTER(POINTER(c_ubyte)), POINTER(c_int), c_bool]

        self.img_pointer = None
        self.khiImageHeight = 0
        self.khiImageWidth = 0
        self.khiScanScale = 0
        self.m_id = ""
        self.control_point = [0, 0]


    # def __del__(self):
    #     self._Objdll_ = []


    def get_slide_pointer(self, filename):
        self.img_pointer = ImageInfoStruct()
        path_buf = ctypes.c_char_p(filename.encode())  # byte array

        return self.InitImageFileFunc(byref(self.img_pointer), path_buf)

    def get_header_info(self):
        khiImageHeight = c_int()
        khiImageWidth = c_int()
        khiScanScale = c_int()
        khiSpendTime = c_float()
        khiScanTime = c_double()
        khiImageCapRes = c_float()
        khiImageBlockSize = c_int()

        success = self.GetHeaderInfoFunc(self.img_pointer, byref(khiImageHeight), byref(khiImageWidth),
                                         byref(khiScanScale),
                                         byref(khiSpendTime), byref(khiScanTime), byref(khiImageCapRes),
                                         byref(khiImageBlockSize))

        self.khiImageHeight = khiImageHeight.value
        self.khiImageWidth = khiImageWidth.value
        self.khiScanScale = khiScanScale.value

        return success

    def open_slide(self, filename, id_string):
        tag = self.get_slide_pointer(filename)
        self.m_id = id_string
        if tag:
            return self.get_header_info()
        return False

    def get_id(self):
        return self.m_id;

    def get_image_width_height_byScale(self, scale):
        if self.khiScanScale == 0:
            return 0, 0

        ImageHeight = np.rint(self.khiImageHeight * scale / self.khiScanScale).astype(np.int64)
        ImageWidth = np.rint(self.khiImageWidth * scale / self.khiScanScale).astype(np.int64)
        return ImageWidth, ImageHeight

    def release_slide_pointer(self):
        # if self.img_pointer :
        return self.UnInitImageFileFunc(byref(self.img_pointer))

    def get_image_block(self, fScale, sp_x, sp_y, nWidth, nHeight, isFile=False):
        pBuffer = POINTER(c_ubyte)()
        DataLength = c_int()
        '''
        bool GetImageDataRoiFunc( ImageInfoStruct sImageInfo, float fScale, 
        int sp_x, int sp_y, int nWidth, int nHeight,
        BYTE** pBuffer, int&DataLength, bool flag);

        参数：
        1.	sImageInfo：传入图像数据指针
        2.	fScale：传入倍率
        3.	sp_x：左上角X坐标
        4.	sp_y：右上角Y坐标
        5.	nWidth：宽度
        6.	nHeight：高度
        7.	pBuffer：返回图像数据指针
        8.	DataLength：返回图像字节长度
        9.	flag：true

        '''
        tag = self.GetImageDataRoiFunc(self.img_pointer, fScale, sp_x, sp_y, nWidth, nHeight, byref(pBuffer),
                                       byref(DataLength), True)
        # print("DataLength = ", DataLength)
        # print(pBuffer)
        data = np.ctypeslib.as_array(
            (ctypes.c_ubyte * DataLength.value).from_address(ctypes.addressof(pBuffer.contents)))

        if isFile:
            return data
        else:
            resultJPG = Image.open(io.BytesIO(data))
            return resultJPG

    '''关于标注的说明：
    1. 使用 FigureType="Polygon" 的曲线来进行区域边界的标记
    2. 不同的 Color属性来区分 良恶性区域（绿色对应良性，蓝色对应恶性）
    3. 为了提高标注的精度，使用多个分段来进行一个区域的标记，
        这里使用Detail属性相同的曲线组成一个区域的边界, A01,
         第一个字母代表区域编号，后面的数字代表连接顺序（顺时针方向）
         
     4. bug：每段曲线只能向一个方向来画, 各段都是顺或逆时针方向来画
    '''
    def read_annotation(self, filename):
        # 使用minidom解析器打开 XML 文档
        fp = open(filename, 'r', encoding="utf-8")
        content = fp.read()
        fp.close()

        content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')

        DOMTree = xml.dom.minidom.parseString(content)
        collection = DOMTree.documentElement

        Regions = collection.getElementsByTagName("Region")

        border_TUMOR = {}
        border_NORMAL = {}
        for Region in Regions:
            if Region.hasAttribute("FigureType"):
                if Region.getAttribute("FigureType") == "Polygon":
                    Vertices = Region.getElementsByTagName("Vertice")
                    posArray = np.zeros((len(Vertices), 2))
                    range_type = int(Region.getAttribute("Color"))
                    contour_id = Region.getAttribute("Detail")

                    i = 0
                    for item in Vertices:
                        posArray[i][0] = float(item.getAttribute("X"))
                        posArray[i][1] = float(item.getAttribute("Y"))
                        i += 1

                    if range_type == TUMOR_RANGE_COLOR:
                        border_TUMOR[contour_id] = posArray
                    elif range_type == NORMAL_RANGE_COLOR:
                        border_NORMAL[contour_id] = posArray

        # merge
        self.ano_TUMOR = []
        self.ano_NORMAL = []

        border_TUMOR = sorted(border_TUMOR.items(), key=lambda x: x[0], reverse=False)
        self.merge_border(border_TUMOR, self.ano_TUMOR)

        border_NORMAL = sorted(border_NORMAL.items(), key=lambda x: x[0], reverse=False)
        self.merge_border(border_NORMAL, self.ano_NORMAL)

        self.read_remark(Regions)
        return

    def read_remark(self, filename):
        # 使用minidom解析器打开 XML 文档
        fp = open(filename, 'r', encoding="utf-8")
        content = fp.read()
        fp.close()

        content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')

        DOMTree = xml.dom.minidom.parseString(content)
        collection = DOMTree.documentElement

        Regions = collection.getElementsByTagName("Region")

        x = 0
        y = 0
        n = 0
        for Region in Regions:
            if Region.hasAttribute("FigureType"):
                if Region.getAttribute("FigureType") == "Line":
                    Vertices = Region.getElementsByTagName("Vertice")
                    for item in Vertices:
                        x += float(item.getAttribute("X"))
                        y += float(item.getAttribute("Y"))
                        n += 1

        if n > 0:
            x = x / n
            y = y / n

        self.control_point = [x, y]
        return x, y

    def merge_border(self, border, result):
        data = []
        pre_id = ''
        for k in border:
            id = k[0][0]
            if len(pre_id) == 0:  # 刚开始，第一段
                data = k[1]
            elif pre_id == id:  # 同一段
                data = np.concatenate((data, k[1]))
            else:  # 新的一段开始
                result.append(data)
                data = k[1]
            pre_id = id

        if len(data) > 0:
            result.append(data)
        return

    def create_mask_image(self, scale=20):
        w, h = self.get_image_width_height_byScale(scale)
        img = np.zeros((h, w), dtype=np.byte)

        for contour in self.ano_TUMOR:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 1

        for contour in self.ano_NORMAL:
            normal_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(normal_range[:, 1], normal_range[:, 0])
            img[rr, cc] = 0

        return img
