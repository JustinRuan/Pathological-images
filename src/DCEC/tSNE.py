import numpy as np
import matplotlib.pyplot as plt
from keras.engine.saving import load_model
from sklearn import manifold
from datasets import load_mnist
from time import time

# digits = datasets.load_digits(n_class=6)
# X, y = digits.data, digits.target

###
# 對mnist數據集使用自編碼器的編碼部分提取特征，並將该多维特征使用t-SNE将至二维，并显示出来。
###
X, y = load_mnist()

t0 = time()
encoder=load_model("H:/zhoujingfan/DCEC-master/DCEC-master/dcec_encode_model_final.h5")
X=encoder.predict(X)
print('MNIST data shape:', X.shape)
t1 = time()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
t2 = time()

#'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()

t3 = time()

print('predict time:  ', t1 - t0)
print('TSNE dimensionality reduction time:', t2 - t1)
print('Visual display time:     ', t3 - t2)