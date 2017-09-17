import numpy as np
import ctypes
from ctypes.wintypes import LPARAM
from ctypes import c_char_p, c_int, c_float, c_double, c_bool, c_ubyte
from ctypes import POINTER, byref
from PIL import Image
import io
import xml.dom.minidom
from skimage import draw

KFB_SDK_PATH = "D:\CloudSpace\DoingNow\WorkSpace\lib\KFB_SDK"
TUMOR_RANGE_COLOR = 4278190335
NORMAL_RANGE_COLOR = 4278222848


# 定义结构体
class ImageInfoStruct(ctypes.Structure):
    _fields_ = [("DataFilePTR", LPARAM)]


class DigitalSlide(object):
    def __init__(self):
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

    # def __del__(self):
    #     if self.img_pointer :
    #         self.img_pointer = None
    #     self = None
    #     return

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

    def open_slide(self, filename):
        tag = self.get_slide_pointer(filename)
        if tag:
            return self.get_header_info()
        return False

    def get_image_width_height_byScale(self, scale):
        if self.khiScanScale == 0:
            return 0, 0

        ImageHeight = np.rint(self.khiImageHeight * scale / self.khiScanScale).astype(np.int64)
        ImageWidth = np.rint(self.khiImageWidth * scale / self.khiScanScale).astype(np.int64)
        return ImageWidth, ImageHeight

    def release_slide_pointer(self):
        # if self.img_pointer :
        return self.UnInitImageFileFunc(byref(self.img_pointer))

    def get_image_block(self, fScale, sp_x, sp_y, nWidth, nHeight):
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

        resultJPG = Image.open(io.BytesIO(data))

        return resultJPG

    def read_annotation(self, filename):
        # 使用minidom解析器打开 XML 文档
        fp = open(filename, 'r', encoding="utf-8")
        content = fp.read()
        fp.close()

        content = content.replace('encoding="gb2312"', 'encoding="UTF-8"')

        DOMTree = xml.dom.minidom.parseString(content)
        collection = DOMTree.documentElement

        Regions = collection.getElementsByTagName("Region")

        self.ano_TUMOR = []
        self.ano_NORMAL = []
        for Region in Regions:
            if Region.hasAttribute("FigureType"):
                if Region.getAttribute("FigureType") == "Polygon":
                    Vertices = Region.getElementsByTagName("Vertice")
                    posArray = np.zeros((len(Vertices), 2))

                    i = 0
                    for item in Vertices:
                        posArray[i][0] = float(item.getAttribute("X"))
                        posArray[i][1] = float(item.getAttribute("Y"))
                        i += 1

                    range_type = int(Region.getAttribute("Color"))
                    if range_type == TUMOR_RANGE_COLOR:
                        self.ano_TUMOR.append(posArray)
                    elif range_type == NORMAL_RANGE_COLOR:
                        self.ano_NORMAL.append(posArray)

        return

    def create_mask_image(self, scale=20):
        w, h = self.get_image_width_height_byScale(scale)
        img = np.zeros((h, w), dtype=np.uint8)

        for contour in self.ano_TUMOR:
            tumor_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(tumor_range[:, 1], tumor_range[:, 0])
            img[rr, cc] = 1

        for contour in self.ano_NORMAL:
            normal_range = np.rint(contour * scale).astype(np.int)
            rr, cc = draw.polygon(normal_range[:, 1], normal_range[:, 0])
            img[rr, cc] = 0

        return img
