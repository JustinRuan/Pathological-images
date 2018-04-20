from PIL import Image, ImageEnhance
from svmutil import *
import utils

# 训练文件路径
train_path = utils.DATA_FILE_PATH + 'train_data.txt'
# 测试文件路径，即待分割图片切成小块之后的文件
test_path = utils.DATA_FILE_PATH + 'seg_data.txt'
# 读入训练文件
y, x = svm_read_problem(train_path)
# 训练svm分类模型(此处先grid求最优参数===>（c,g）)
m=svm_train(y,x,'-c 2.0 -g 0.03125 -b 1')
# 读取待分割图片
ori_image = Image.open('../data/image_data/2.jpg')
# 获取图片宽、高
ImageHeight = ori_image.height
ImageWidth = ori_image.width
# 直接读取libsvm格式的文件，返回分类标签([])和数据([[]])
y2,x2 = svm_read_problem(test_path)

# 初始化分割后的图片
i = 0
# 同样用112*112的窗口以8的步长滑动切割大幅图像，得到样本，进行检测
for y in range(0, ImageHeight, 8):
    for x in range(0, ImageWidth, 8):
        region = (x, y, x + 112, y + 112)
        cropimg = ori_image.crop(region)
        # svm分类，p_val的值是预测概率估计值
        p_lable,p_acc, p_val = svm_predict(y2[i:i + 1], x2[i:i + 1], m, options='-b 1')
        # 若预测类别为正，则增强该区域的色度
        # 由于样本之间有重叠区域，经色度叠加可显著凸显癌细胞区域
        # （除色度叠加外可寻找更好的图像处理方法。。。）
        if p_lable == [2.0]:
            enh_bri = ImageEnhance.Color(cropimg)
            brightness = 1.00002
            image_brightened = enh_bri.enhance(brightness)
            ori_image.paste(image_brightened, region)
        i += 1
        print(i)

ori_image.show()