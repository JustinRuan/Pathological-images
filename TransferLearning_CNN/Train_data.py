import caffe
import datetime
import numpy as np
from PIL import Image
import os
import utils

# 根据diff100制作SVM训练集

# 路径设置
# 分类deploy
deploy =utils.MODEL_PATH + '/alexnet/deploy.prototxt'
# 训练好的模型
caffe_model = utils.MODEL_PATH + '/Accurate_model/alexnet_iter_4000.caffemodel'
# 训练正样本图片路径
path_positive = utils.PATH_TRAIN_IMAGE + 'cancer/'
# 训练负样本图片路径
path_negative = utils.PATH_TRAIN_IMAGE + 'normal/'
# 训练集存储路径
path_train_svm = utils.DATA_FILE_PATH + 'train_data.txt'
# 已生成的diff100路径
path_diff100 = utils.DATA_FILE_PATH + 'diff100.txt'

# 调用网络，设置GPU模式
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)

#加载均值文件
mu = np.load('H:/PYPROJECT/Workspace/Pathological-images/DetectCancer/models/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
#数据预处理
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

#特征提取，并计算所有样本在第K个dimension上的均值
def extract(path):
    array_total = np.zeros(4096)  # 初始化一个全零的数组，用于后面多张图片向量的拼接（第一行全0，第i行是第i张图片的特征向量）
    Num = 0 #样本总量
    for files in os.listdir(path):
        Num += 1
        image = Image.open(path + files)
        patch = np.array(image)
        net.blobs['data'].data[...] = transformer.preprocess('data', patch)
        net.forward()
        #Vec为网络fc7层对单张样本的输出，为4096长度的数组
        Vec = net.blobs['fc7'].data[0]
        # 每遍历一张图片，将该图片的特征向量拼接到下一行
        array_total = np.vstack((array_total, Vec))
    # 将数组转化为矩阵
    mat_total = np.mat(array_total)
    # 求矩阵每一列的和，得到一个1*4096的矩阵
    sum_col = mat_total.sum(axis=0)
    print(Num)
    return array_total

def create_txt(label_p, label_n, path_diff100,path, path_positive, path_negative):
    diff100_temp = np.loadtxt(path_diff100)
    # 转格式
    diff100 = diff100_temp.astype(int)
    # 正样本的特征向量集
    array_total_p = extract(path_positive)
    # 负样本的特征向量集
    array_total_n = extract(path_negative)

    # 开始写入SVM训练集
    f = open(path, 'w+')
    for j in range(1, len(array_total_p)):
        f.write("%i "%label_p)
        for i in range(100):
            vec_temp = array_total_p[j]
            newcontext = "%i:"%(i+1) + "%f"%(vec_temp[diff100[i]]) + " "
            f.write(newcontext)
        f.write('\n')
    for j in range(1, len(array_total_n)):
        f.write("%i "%label_n)
        for i in range(100):
            vec_temp = array_total_n[j]
            newcontext = "%i:"%(i+1) + "%f"%(vec_temp[diff100[i]]) + " "
            f.write(newcontext)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    starttime = datetime.datetime.now()

    create_txt(2, 1, path_diff100, path_train_svm, path_positive, path_negative)

    endtime = datetime.datetime.now()
    print("program running for %d seconds"%(endtime - starttime).seconds)

