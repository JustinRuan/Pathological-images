import caffe
import datetime
import numpy as np
from PIL import Image
import os
import utils

# 用已经训练完毕的卷积神经网络模型寻找正负样本差异最大的100个特征的特征位点(注意是位点)
# 记录为diff100.txt
# 后续实验中提取正负样本特征值 制作SVM训练集时需运用到diff100

# 路径设置
# 网络结构deploy.prototxt
deploy =utils.MODEL_PATH + '/alexnet/deploy.prototxt'
# 训练好的模型
caffe_model = utils.MODEL_PATH + '/Accurate_model/alexnet_iter_4000.caffemodel'
# 正样本路径
path_positive = utils.PATH_TRAIN_IMAGE + 'cancer/'
# 负样本路径
path_negative = utils.PATH_TRAIN_IMAGE + 'normal/'
# 特征位点保存路径
path_diff100 = utils.DATA_FILE_PATH + 'diff100.txt'

# 调用网络，设置GPU模式
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)

# 加载均值文件
mu = np.load('H:/PYPROJECT/Workspace/Pathological-images/DetectCancer/models/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
# 数据预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# 设置数据输入层参数，批处理量为50，三通道
# 此处若为AlexNet则输如reshape为227*227，若为GoogLeNet则reshape为224*224
net.blobs['data'].reshape(50,
                            3,
                            227, 227)

#网络结构和各层输出量
print("网 络 结 构：")
for layer_name, blob in net.blobs.items():
    print(layer_name + '\t' + str(blob.data.shape))

print('=================================')

print("各 层 输 出：")
for layer_name, param in net.params.items():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

# 特征提取，并计算所有样本在第K个dimension上的均值
# 输入样本后，提取AlexNet的fc7层的输出，该层对于单张样本数，输出位4096长度的数组，代表4096个维度(dimension)的特征
# GoogLeNet的loss3/classifier层的输出，为1000....
def extract(path):
    # 初始化一个全零的数组，用于后面多张图片向量的拼接（第一行全0，第i行是第i张图片的特征向量）
    array_total = np.zeros(4096)
    # 样本总量
    Num = 0
    for files in os.listdir(path):
        Num += 1
        image = Image.open(path + files)
        patch = np.array(image)
        net.blobs['data'].data[...] = transformer.preprocess('data', patch)
        net.forward()
        #Vec为网络fc7层对单张样本的输出，为4096长度的数组
        Vec = net.blobs['fc7'].data[0]
        # 每处理一张图片，将该图片的特征向量拼接到下一行
        array_total = np.vstack((array_total, Vec))
    # 将数组转化为矩阵
    mat_total = np.mat(array_total)
    # 求矩阵每一列的和，得到一个1*4096的矩阵
    sum_col = mat_total.sum(axis=0)
    print(Num)
    # 求得样本在各个维度的均值
    AVG = sum_col / Num
    return AVG, array_total


if __name__ == '__main__':
    # 建立数组diffk，存储正负样本在维度k处的特征值差值
    diffk = []
    diff100 = []
    starttime = datetime.datetime.now()

    AVG_p, array_total_p = extract(path_positive)
    AVG_n, array_total_n = extract(path_negative)
    # diff 是一个矩阵matrix
    diff = abs(AVG_p - AVG_n)
    for k in range(0,4095):
        diffk.append(diff[0,k])
    # 排序并翻转，记录索引
    diffk = sorted(range(len(diffk)),reverse=True,key=diffk.__getitem__)


    # 写入TXT
    f = open(path_diff100, 'w+')
    for k in range(100):
        #f.write(str(diff_selected[k]))
        f.write(str(diffk[k]))
        f.write(' ')
    f.close()

    endtime = datetime.datetime.now()
    print("program running for %d seconds"%(endtime - starttime).seconds)


