import math
from skimage import io, color, img_as_float
import numpy as np
from connectivity import enforce_connectivity
from sklearn.cluster import KMeans, AgglomerativeClustering
from skimage.segmentation._slic import _enforce_label_connectivity_cython

# 聚类中心类
class Cluster(object):
    cluster_index = 1

    def __init__(self, h=None, w=None, data=None):
        self.update(h, w, data=data)
        self.pixels = []  # 该聚类中心包含的像素点(h, w)
        self.no = self.cluster_index   # 聚类中心标签号
        Cluster.cluster_index += 1

    def update(self, h, w, data):
        self.h = h
        self.w = w
        self.data = np.array(data)


# SLIC 类
class SLICProcessor(object):
    def __init__(self, data, K, M):
        self.K = K
        self.M = M

        self.data = np.array(data)
        # 计算step的值S
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.feature_dim = self.data.shape[2]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
        # 设立全局的聚类中心列表，标签，距离
        self.clusters = []  # 聚类中心类的对象的列表
        self.label = {}     # 用字典的形式(h, w)保存每一个点的聚类中心,  键：(h, w)，值：cluster聚类中心类的一个对象
        # self.label_img = np.full((self.image_height, self.image_width), -1)
        self.dis = np.full((self.image_height, self.image_width), np.inf)  # np.inf 指的是无穷大

# 初始化聚类中心
    # 生成聚类中心类的对象
    def make_cluster(self, h, w):
        return Cluster(h, w, self.data[h][w])

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(int(h),int(w)))
                w += self.S
            w = self.S / 2
            h += self.S

# 这也属于初始化聚类中心的改进，将从一个3*3 的矩阵中找一个梯度最小的点，作为于初始化的聚类中心
#     def move_clusters(self):
#         for cluster in self.clusters:
#             if cluster.h + 1 >= self.image_height or cluster.w + 1 >= self.image_width:
#                 if cluster.w + 1 >= self.image_width:
#                     cluster.w = self.image_width - 2
#                 if cluster.h + 1 >= self.image_height:
#                     cluster.h = self.image_height - 2
#                 cluster.update(cluster.h, cluster.w, self.data[cluster.h][cluster.w])
#
#             cluster_gradient = self.get_gradient(cluster.h, cluster.w)
#             for dh in range(-1, 2):
#                 for dw in range(-1, 2):
#                     _h = cluster.h + dh
#                     _w = cluster.w + dw
#                     if 1 < _w < self.image_width - 1 and 1 < _h < self.image_height - 1:
#                         new_gradient = self.get_gradient(_h, _w)
#                         if new_gradient < cluster_gradient:
#                             cluster.update(_h, _w, self.data[_h][_w])
#                             cluster_gradient = new_gradient
    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w])
                        cluster_gradient = new_gradient

    # 计算一个点的梯度
    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        #          top
        # left    x=(w, h)    right
        #         bottom
        # data.shape = (h, w, 32)

        right_bottom = self.data[h + 1, w + 1, :]
        x = self.data[h, w, :]
        gradient = np.sum(right_bottom - x)
        return gradient

    def features_similarity(self, f1, f2):
        # 欧氏距离
        # result = math.sqrt(np.sum(np.power(f1 - f2, 2)))

        #Cosine
        d = np.dot(f1, f2)/(np.linalg.norm(f1) * np.linalg.norm(f2))
        result = math.exp(1 - d)

        return result


# 计算聚类中心2S范围的点距离聚类中心的距离，并以此更新labels，dis
    def assignment(self):
        for cluster in self.clusters:
            for h in range(int(cluster.h - 2 * self.S), int(cluster.h + 2 * self.S)):
                if h < 0 or h >= self.image_height: continue  # 越界
                for w in range(int(cluster.w - 2 * self.S), int(cluster.w + 2 * self.S)):
                    if w < 0 or w >= self.image_width: continue  # 越界
                    # 每一个聚类中心点的维度数，一般为3
                    feature = self.data[h,w,:]  # 图像中的一个点
                    # 特征距离
                    Dc = self.features_similarity(feature, cluster.data)
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))   # 空间距离
                    # Ds = math.pow(h - cluster.h, 2) + math.pow(w - cluster.w, 2)
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))   # 最终距离
                    # D = Dc / math.pow(self.M, 2) + Ds / math.pow(self.S, 2)
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))  # 聚类中心类中添加一个像素点的位置
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D
                        # if(self.label[h][w]==-1):
                        #    self.label[h][w]=cluster.no
                        #    cluster.pixels.append((h, w))
                        # else:
                        #    #  从旧的聚类中心删掉改点，在新的加上该点
                        #    for oldCluster in self.clusters:
                        #        if self.label[h][w]==oldCluster.no:
                        #            oldCluster.pixels.remove((h, w))
                        #            break
                        #    self.label[h][w]=cluster.no
                        #    cluster.pixels.append((h, w))
                        # self.dis[h][w] = D


# 更新聚类中心
    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = 0.0
            sum_f = np.zeros((self.feature_dim))

            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                sum_f += self.data[p[0], p[1], :]

            number = len(cluster.pixels)
            _h = sum_h / number
            _w = sum_w / number

            data = sum_f / number
            cluster.update(_h, _w, data)

    def clusting(self, iter_num = 10, enforce_connectivity = True, min_size_factor = 0.5, max_size_factor = 3.0):
        self.init_clusters()            # 初始化聚类中心
        self.move_clusters()            # 移动初始化的聚类中心到梯度最小点去,作用不大
        for i in range(iter_num):
            self.assignment()           # 计算聚类中心2S范围的点距离聚类中心的距离
            self.update_cluster()       # 更新聚类中心
            print("iter_{}".format(i))

        label_img = np.full((self.image_height, self.image_width), -1)
        for (h, w), cluster in self.label.items():
            label_img[h, w] = cluster.no

        if enforce_connectivity:
            segment_size = self.feature_dim * self.image_height * self.image_width / self.K
            min_size = int(min_size_factor * segment_size)
            max_size = int(max_size_factor * segment_size)
            # label_img = _enforce_label_connectivity_cython(label_img[...,np.newaxis],
            #                                             min_size,
            #                                             max_size)
            label_img = self.enforce_connectivity(label_img)

        return label_img


# # 迭代10次
#     def iterate_10times(self):
#         self.init_clusters()            # 初始化聚类中心
#         self.move_clusters()            # 移动初始化的聚类中心到梯度最小点去,作用不大
#         for i in range(10):
#             self.assignment()           # 计算聚类中心2S范围的点距离聚类中心的距离
#             self.update_cluster()       # 更新聚类中心
#             print("iter_{}".format(i))
#         return self.label

#强连接，合并孤立点
    def enforce_connectivity(self, label_img):
        label = label_img[np.newaxis, :, :]
        label = label.astype(np.int64)
        # 调用cython来合并孤立点
        label = enforce_connectivity(label, int(self.S * self.S * 0.2), int(self.S * self.S * 3))
        label = np.squeeze(label, axis=(0,))
        print("合并孤立点后的label数量{}".format(len(np.unique(label))))

        # 遍历self.label 改变 self.cluster.pixels
        self.clusters=[]
        Cluster.cluster_index=0
        for i in range(len(np.unique(label))):
            self.clusters.append(Cluster())
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                for cluster in self.clusters:
                    if cluster.no == label[x][y]:
                        cluster.pixels.append((x, y))
                        break
        self.update_cluster()

        return label



    # # 对聚类中心进行kmeans聚类
    # def cluster_kmeans(self,K):
    #     cluster_data = []
    #     for cluster in self.clusters:
    #         cluster_data.append(cluster.data)
    #     kmeans = KMeans(n_clusters=K, random_state=0).fit(cluster_data)
    #     i=0
    #     label = np.full((self.image_height, self.image_width), -1)
    #     for cluster in self.clusters:
    #         # cluster.no=kmeans.labels_[i]
    #         for pixel in cluster.pixels:
    #             label[pixel[0],pixel[1]]=kmeans.labels_[i]
    #         #####这里没有更新维护聚类中心类以及聚类中心类包含的点#
    #         i = i + 1
    #     self.label = label
    #
    #     return self.label
    #
    # # 对聚类中心进行Hierarchical聚类，层次聚类
    # def cluster_Hierarchical(self, K):
    #     cluster_data = []
    #     for cluster in self.clusters:
    #         cluster_data.append(cluster.data)
    #     centerCluster = AgglomerativeClustering(n_clusters=K).fit(cluster_data)
    #     i = 0
    #     label = np.full((self.image_height, self.image_width), -1)
    #     for cluster in self.clusters:
    #         # cluster.no=kmeans.labels_[i]
    #         for pixel in cluster.pixels:
    #             label[pixel[0], pixel[1]] = centerCluster.labels_[i]
    #         i = i + 1
    #     self.label = label
    #     return self.label

# if __name__ == '__main__':
#
#     p = SLICProcessor("133145.jpg", 2, 10)     #实例化一个SLICPro 对象，传入图像，k，m
#     p.iterate_10times()