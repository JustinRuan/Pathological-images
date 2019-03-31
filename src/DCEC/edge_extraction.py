from keras.datasets import cifar10
import numpy as np
from skimage.color import rgb2gray, rgb2grey
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.exposure import rescale_intensity
from skimage import data,filters

###
# 使用sobel算子提取cifar10图像的边缘
###

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
y = np.squeeze(y)
x = x.reshape((x.shape[0],32,32,3))
# x = np.divide(x, 255.)  # normalize as it does in DEC paper
print ('cifar10 samples', x.shape)
print('cifar10 label samples', y.shape)

x_test = x[:100]
x_gray_test = np.zeros(shape=(100,32,32))
n=0
for i in x_test:
    i_gray = rgb2gray(i)
    i_edges = filters.sobel(i_gray)
    i_edges = i_edges.reshape(i_edges.shape[0], i_edges.shape[1]).astype('float32')
    x_gray_test[n] = i_edges
    n+=1

print(x_gray_test.shape)

def hstackimgs(min, max, images):
    return np.hstack(images[i] for i in range(min, max))

def sqstackimgs(length, height, images):
    return np.vstack(hstackimgs(i*length, (i+1)*length, images) for i in range(height))

def sbscompare(images1, images2, length, height):
    A = sqstackimgs(length, height, images1)
    # B = sqstackimgs(length, height, images2)
    # C = np.ones((A.shape[0], 32, 3)).astype('uint8')  # 因为这里的C是浮点数，所以会使得np.hstack((A, C, B))变为浮点数
    # return np.hstack((A, C, B))
    return A

import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(sbscompare(x_test, x_test, 10, 10))
plt.subplot(122)
plt.imshow(sbscompare(x_gray_test, x_test, 10, 10))
plt.axis('off')
plt.rcParams["figure.figsize"] = [60,60]
plt.show()



# @adapt_rgb(each_channel)
# def sobel_each(image):
#     return filters.sobel(image)
#
# @adapt_rgb(hsv_value)
# def sobel_hsv(image):
#     return filters.sobel(image)


###
# from skimage import data,filters
# camera = data.camera()
# edges = filters.sobel(camera)
# print(edges.shape)
# edges1 = feature.canny(camera)
# edges2 = feature.canny(camera, sigma=3)

###
# from skimage import data,filters,feature
# astronaut = data.astronaut()
# astronaut = rgb2gray(astronaut)
# print(astronaut.shape)
# astronaut_edge = filters.sobel(astronaut)
# astronaut_edge = astronaut_edge.reshape(astronaut_edge.shape[0], astronaut_edge.shape[1]).astype('float32')
# astronaut_edge2= feature.canny(astronaut)
# astronaut_edge3 = feature.canny(astronaut, sigma=3)
#
# import matplotlib.pyplot as plt
# plt.subplot(221)
# plt.imshow(astronaut)
# plt.subplot(222)
# plt.imshow(astronaut_edge)
# plt.subplot(223)
# plt.imshow(astronaut_edge2)
# plt.subplot(224)
# plt.imshow(astronaut_edge3)
# plt.show()