import caffe
import numpy as np
from PIL import Image
import utils

#分类deploy
deploy =utils.MODEL_PATH + '/alexnet/deploy.prototxt'
#训练好的模型
caffe_model = utils.MODEL_PATH + '/Accurate_model/alexnet_iter_4000.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)

# 设置数据输入层参数，批处理量为50，三通道
# 此处若为AlexNet则输如reshape为227*227，若为GoogLeNet则reshape为224*224
net.blobs['data'].reshape(50,
                                  3,
                                  227, 227)

# 加载均值文件
mu = np.load('H:/PYPROJECT/Workspace/Pathological-images/DetectCancer/models/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
# 数据预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))


#待分割图片
fullImage = Image.open('../data/image_data/2.jpg')
height = fullImage.height
width = fullImage.width


diff100_temp = np.loadtxt(utils.DATA_FILE_PATH + 'diff100.txt')
diff100 = diff100_temp.astype(int)
f = open(utils.DATA_FILE_PATH + 'seg_data.txt', 'w+')
# 设定大小为112*112的窗口，以8的步长滑动切割大幅图像
for y in range(0, height, 8):
    for x in range(0, width, 8):
        region = (x, y, x + 112, y + 112)
        # 图像剪裁
        cropimg = fullImage.crop(region)
        patch = np.array(cropimg)

        net.blobs['data'].data[...] = transformer.preprocess('data', patch)
        net.forward()
        Vec = net.blobs['fc7'].data[0]
        # 为符合输入格式故 write("1 ")
        f.write("1 ")
        for i in range(0,100):
            f.write("%i:"%(i + 1) + "%f"%(Vec[diff100[i]]) + " ")
        f.write('\n')



