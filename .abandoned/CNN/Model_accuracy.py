import caffe
import numpy as np
from PIL import Image
import os
import shutil
import utils

deploy = utils.MODEL_PATH + 'alexnet/deploy.prototxt'
# 训练好的caffemodel
caffe_model = utils.MODEL_PATH + 'Accurate_model/alexnet_iter_4000.caffemodel'
# 样本测试集
path = utils.PATH_TEST_IMAGE + 'normal/'
# 难样本复制地址
HS_path = HS_path = utils.HS_PATH + 'cancer/'

def accuracy_test():

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(deploy, caffe_model, caffe.TEST)
    # 设定图片的shape格式(1,3,28,28)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_transpose('data', (2, 0, 1))
    # 减去均值，若训练时未用到均值文件，则不需要此步骤
    # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    # 缩放到【0，255】之间
    transformer.set_raw_scale('data', 255)
    # 交换通道，将图片由RGB变为BGR
    transformer.set_channel_swap('data', (2, 1, 0))

    i = 0
    j = 0
    for files in os.listdir(path):
        image = Image.open(path+files)
        #样本矩阵化
        patch = np.array(image)
        net.blobs['data'].data[...] = transformer.preprocess('data', patch)
        net.forward()
        output = net.forward()
        output_prob = output['prob'][0]
        #样本类别的输出概率值
        prob = net.blobs['prob'].data[0].flatten()
        #提取难样本
        if(abs(prob[0] - prob[1]) < 0.3):
            shutil.copyfile(path + files, HS_path + files)
        #样本识别率大于85%则分类正确
        if prob[0] > 0.85:
            i += 1
        if prob[1] > 0.85:
            j += 1

        print(prob[0],end=',')
        print(prob[1])


    print("normal_patches: %d" %i + '\n' + "cancer_patches: %d" %j)
    #测试数据集为正样本时
    accuracy = j/(i+j)*100
    #测试数据集为负样本时
    # accuracy = i/(i+j)*100

    print("accuracy: %d" %accuracy + "%")

if __name__ == '__main__':
    accuracy_test()