#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Justin'
__mtime__ = '2018-10-13'

"""

import numpy as np
import ctypes
from ctypes.wintypes import LPARAM
from ctypes import c_char_p, c_int, c_float, c_double, c_bool, c_ubyte
from ctypes import POINTER, byref
from PIL import Image
import io
import xml.dom.minidom

TUMOR_RANGE_COLOR = 4278190335
NORMAL_RANGE_COLOR = 4278222848


# 定义结构体
class ImageInfoStruct(ctypes.Structure):
    _fields_ = [("DataFilePTR", LPARAM)]


class Slice(object):
    def __init__(self, SDK_path):
        '''
        初始化过程
        :param SDK_path: SDK 的所在路径
        '''
        # KFB_SDK_PATH = "D:/CloudSpace/DoingNow/WorkSpace/lib/KFB_SDK"
        KFB_SDK_PATH = SDK_path
        self._Objdll_ = ctypes.windll.LoadLibrary(KFB_SDK_PATH + "/x64/ImageOperationLib")

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

    # def __del__(self):
    #     if self.img_pointer :
    #         self.img_pointer = None
    #     self = None
    #     return

    def get_slide_pointer(self, filename):
        '''
        得到切片指针
        :param filename: 切片文件路径
        :return:
        '''
        self.img_pointer = ImageInfoStruct()
        path_buf = ctypes.c_char_p(filename.encode())  # byte array

        return self.InitImageFileFunc(byref(self.img_pointer), path_buf)

    def get_header_info(self):
        '''
        读文件的头信息
        :return: bool 是否打开
        '''
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
        '''
        打开切片文件
        :param filename: 切片文件
        :param id_string: 切片编号
        :return: 是否成功打开
        '''
        tag = self.get_slide_pointer(filename)
        self._id = id_string
        if tag:
            return self.get_header_info()
        return False

    def get_id(self):
        '''
        得取切片的编号
        :return: 返回切片编号
        '''
        return self._id;

    def get_image_width_height_byScale(self, scale):
        '''
        得到指定倍镜下，整个切片图像的大小
        :param scale: 提取图像所在的倍镜数
        :return: 图像的大小，宽（x）高（y）
        '''
        if self.khiScanScale == 0:
            return 0, 0

        ImageHeight = np.rint(self.khiImageHeight * scale / self.khiScanScale).astype(np.int64)
        ImageWidth = np.rint(self.khiImageWidth * scale / self.khiScanScale).astype(np.int64)
        return ImageWidth, ImageHeight

    def release_slide_pointer(self):
        # if self.img_pointer :
        return self.UnInitImageFileFunc(byref(self.img_pointer))

    def get_image_block_file(self, fScale, c_x, c_y, nWidth, nHeight):
        '''
        提取所在位置的图块文件流
        :param fScale: 所使用倍镜数
        :param c_x: 中心x坐标
        :param c_y: 中心y坐标
        :param nWidth: 图块的宽
        :param nHeight: 图块的高
        :return: 图块的文件流，保存它就成为JPG文件
        '''
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
        #从中心坐标移动到左上角坐标
        sp_x = c_x - (nWidth >> 1)
        sp_y = c_y - (nHeight >> 1)

        tag = self.GetImageDataRoiFunc(self.img_pointer, fScale, sp_x, sp_y, nWidth, nHeight, byref(pBuffer),
                                       byref(DataLength), True)
        data = np.ctypeslib.as_array(
            (ctypes.c_ubyte * DataLength.value).from_address(ctypes.addressof(pBuffer.contents)))
        return data

    def get_image_block(self, fScale, c_x, c_y, nWidth, nHeight):
        '''
        提取指定位置的图块
        :param fScale: 所使用倍镜数
        :param c_x: 中心x坐标
        :param c_y: 中心y坐标
        :param nWidth: 图块的宽
        :param nHeight: 图块的高
        :return: 图块的矩阵，用于算法的处理
        '''
        data = self.get_image_block_file(fScale, c_x, c_y, nWidth, nHeight)
        return Image.open(io.BytesIO(data))

        # if isFile:
        #     return data
        # else:
        #     resultJPG = Image.open(io.BytesIO(data))
        #     return resultJPG

    ############################## v.03 代码 #################################
    '''
    关于标注的说明：
    1. 使用 FigureType="Polygon" 的曲线来进行区域边界的标记
    2. 不同的 Color属性来区分 良恶性区域（绿色对应良性，蓝色对应恶性）
    3. 每个区域用一段封闭曲线进行标注。
    '''
    def read_annotation(self, filename):
        '''
        读取标注文件
        :param filename: 切片标注文件名
        :return:
        '''
        self.ano_TUMOR = []
        self.ano_NORMAL = []

        # 使用minidom解析器打开 XML 文档
        fp = open(filename, 'r', encoding="utf-8")
        content = fp.read()
        fp.close()

        content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')

        DOMTree = xml.dom.minidom.parseString(content)
        collection = DOMTree.documentElement

        Regions = collection.getElementsByTagName("Region")

        for Region in Regions:
            if Region.hasAttribute("FigureType"):
                if Region.getAttribute("FigureType") == "Polygon" :
                    Vertices = Region.getElementsByTagName("Vertice")
                    range_type = int(Region.getAttribute("Color"))
                    # contour_id = Region.getAttribute("Detail")

                    posArray = np.zeros((len(Vertices), 2))
                    i = 0
                    for item in Vertices:
                        posArray[i][0] = float(item.getAttribute("X"))
                        posArray[i][1] = float(item.getAttribute("Y"))
                        i += 1

                    if range_type == TUMOR_RANGE_COLOR:
                        self.ano_TUMOR.append(posArray)
                    elif range_type == NORMAL_RANGE_COLOR:
                        self.ano_NORMAL.append(posArray)

        return

    ############################## v.02 代码 #################################
    '''关于标注的说明：
    1. 使用 FigureType="Polygon" 或 FigureType="Rectangle" 的曲线来进行区域边界的标记
    2. 不同的 Color属性来区分 良恶性区域（绿色对应良性，蓝色对应恶性）
    3. 为了提高标注的精度，标注的改进思路：两级精度的标注
        对于癌变区域和普通区域分成两级标注。每个标注只对应一段封闭曲线。
        
        癌变精标区：在高分辨率（X10~X20）下，进行区域的精确标注。
        癌变粗标区：在低分辨率（X2~X5）下，进行较大范围地标记。
        精标区 用Detail="A"表示。
        粗标区 用Detail="R"表示。
        
        癌变精标区代号 TA， ano_TUMOR_A，在标记中直接给出。
        癌变粗标区代号 TR， ano_TUMOR_R，最终的区域从标记中计算得出， 
                            ano_TUMOR_R = 标记的TR - NR，即在CR中排除NR区域
        正常精标区代号 NA， ano_NORMAL_A，在标记中直接给出。
        正常粗标区代号 NR， ano_NORMAL_R，ano_NORMAL_R = ALL有效区域 - 最终CR
    '''
    # def read_annotation(self, filename):
    #     '''
    #     读取标注文件
    #     :param filename: 切片标注文件名
    #     :return:
    #     '''
    #     self.ano_TUMOR_A = []
    #     self.ano_TUMOR_R = []
    #     self.ano_NORMAL_A = []
    #     self.ano_NORMAL_R = []
    #
    #     # 使用minidom解析器打开 XML 文档
    #     fp = open(filename, 'r', encoding="utf-8")
    #     content = fp.read()
    #     fp.close()
    #
    #     content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')
    #
    #     DOMTree = xml.dom.minidom.parseString(content)
    #     collection = DOMTree.documentElement
    #
    #     Regions = collection.getElementsByTagName("Region")
    #
    #     for Region in Regions:
    #         if Region.hasAttribute("FigureType"):
    #             if Region.getAttribute("FigureType") == "Polygon" or \
    #                     Region.getAttribute("FigureType") == "Rectangle":
    #                 Vertices = Region.getElementsByTagName("Vertice")
    #                 range_type = int(Region.getAttribute("Color"))
    #                 contour_id = Region.getAttribute("Detail")
    #
    #                 if Region.getAttribute("FigureType") == "Polygon":
    #                     posArray = np.zeros((len(Vertices), 2))
    #                     i = 0
    #                     for item in Vertices:
    #                         posArray[i][0] = float(item.getAttribute("X"))
    #                         posArray[i][1] = float(item.getAttribute("Y"))
    #                         i += 1
    #                 else:  # "Rectangle"
    #                     x1 = float(Vertices[0].getAttribute("X"))
    #                     y1 = float(Vertices[0].getAttribute("Y"))
    #                     x2 = float(Vertices[1].getAttribute("X"))
    #                     y2 = float(Vertices[1].getAttribute("Y"))
    #
    #                     # posArray = np.zeros(4, 2)
    #                     posArray = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    #
    #                 if range_type == TUMOR_RANGE_COLOR:
    #                     if contour_id == 'A':
    #                         self.ano_TUMOR_A.append(posArray)
    #                     else:  # contour_id == 'R'
    #                         self.ano_TUMOR_R.append(posArray)
    #                 elif range_type == NORMAL_RANGE_COLOR:
    #                     if contour_id == 'A':
    #                         self.ano_NORMAL_A.append(posArray)
    #                     else:  # contour_id == 'R'
    #                         self.ano_NORMAL_R.append(posArray)
    #
    #     return

    ############################## v.01 代码 #################################
    '''关于标注的说明：
    1. 使用 FigureType="Polygon" 的曲线来进行区域边界的标记
    2. 不同的 Color属性来区分 良恶性区域（绿色对应良性，蓝色对应恶性）
    3. 为了提高标注的精度，使用多个分段来进行一个区域的标记，
        这里使用Detail属性相同的曲线组成一个区域的边界, A01,
         第一个字母代表区域编号，后面的数字代表连接顺序（顺时针方向）

     4. bug：每段曲线只能向一个方向来画, 各段都是顺或逆时针方向来画
     '''
    # def read_annotation(self, filename):
    #     # 使用minidom解析器打开 XML 文档
    #     fp = open(filename, 'r', encoding="utf-8")
    #     content = fp.read()
    #     fp.close()
    #
    #     content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')
    #
    #     DOMTree = xml.dom.minidom.parseString(content)
    #     collection = DOMTree.documentElement
    #
    #     Regions = collection.getElementsByTagName("Region")
    #
    #     border_TUMOR = {}
    #     border_NORMAL = {}
    #     for Region in Regions:
    #         if Region.hasAttribute("FigureType"):
    #             if Region.getAttribute("FigureType") == "Polygon":
    #                 Vertices = Region.getElementsByTagName("Vertice")
    #                 posArray = np.zeros((len(Vertices), 2))
    #                 range_type = int(Region.getAttribute("Color"))
    #                 contour_id = Region.getAttribute("Detail")
    #
    #                 i = 0
    #                 for item in Vertices:
    #                     posArray[i][0] = float(item.getAttribute("X"))
    #                     posArray[i][1] = float(item.getAttribute("Y"))
    #                     i += 1
    #
    #                 if range_type == TUMOR_RANGE_COLOR:
    #                     border_TUMOR[contour_id] = posArray
    #                 elif range_type == NORMAL_RANGE_COLOR:
    #                     border_NORMAL[contour_id] = posArray
    #
    #     # merge
    #     self.ano_TUMOR = []
    #     self.ano_NORMAL = []
    #
    #     border_TUMOR = sorted(border_TUMOR.items(), key=lambda x: x[0], reverse=False)
    #     self.merge_border(border_TUMOR, self.ano_TUMOR)
    #
    #     border_NORMAL = sorted(border_NORMAL.items(), key=lambda x: x[0], reverse=False)
    #     self.merge_border(border_NORMAL, self.ano_NORMAL)
    #     return
    #
    # def merge_border(self, border, result):
    #     data = []
    #     pre_id = ''
    #     for k in border:
    #         id = k[0][0]
    #         if len(pre_id) == 0:  # 刚开始，第一段
    #             data = k[1]
    #         elif pre_id == id:  # 同一段
    #             data = np.concatenate((data, k[1]))
    #         else:  # 新的一段开始
    #             result.append(data)
    #             data = k[1]
    #         pre_id = id
    #
    #     if len(data) > 0:
    #         result.append(data)
    #     return
    #
