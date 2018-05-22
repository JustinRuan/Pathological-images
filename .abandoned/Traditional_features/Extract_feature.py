from skimage import feature
from sklearn.decomposition import PCA
from PIL import Image
from pylab import *
import os
import utils

# 提取特征的保存路径
#正样本路径
path_positive = utils.PATH_TRAIN_IMAGE + 'cancer/'
#负样本路径
path_negative = utils.PATH_TRAIN_IMAGE + 'normal/'
#生成的libsvm格式的传统特征txt路径
path_svm = utils.PATH_SVM + 'texture_feature.txt'

# 提取 glcm 特征,并将特征写入txt文件中
def extract_glcm_feature(path_image):
    #存储所有样本的glcm特征
    feature_glcm_total = []
    for files in os.listdir(path_image):
        #存储单张图片的glcm特征
        textural_feature = []
        #以灰度模式读取图片
        image = array(Image.open(path_image + files).convert('L'))
        # 计算灰度共生矩阵
        glcm = feature.greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
        # 得到不同统计量
        textural_feature.append(feature.greycoprops(glcm, 'contrast')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'dissimilarity')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'homogeneity')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'ASM')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'energy')[0, 0])
        textural_feature.append(feature.greycoprops(glcm, 'correlation')[0, 0])
        # 每遍历一张图片，将该图片的特征向量拼接到下一行
        feature_glcm_total.append(textural_feature)
    # 归一化
    # textural_feature_nor = MaxMinNormalization(feature_glcm_total)
    return feature_glcm_total

# 提取 lbp 特征
def extract_lbp_feature(path_image):
    radius = 1;
    n_point = radius * 8;
    textural_feature_total = []
    for files in os.listdir(path_image):
        image = array(Image.open(path_image + files).convert('L'))
        # thresh = threshold_otsu(image)
        # binary = image > thresh
        # 计算lbp特征
        lbp_feature = feature.local_binary_pattern(image, n_point, radius)
        # 统计直方图
        lbp_hist = np.histogram(lbp_feature, bins=256)
        # 每遍历一张图片，将该图片的特征向量添加到list后面
        textural_feature_total.append(lbp_hist[0])
    # 归一化
    # textural_feature_nor = MaxMinNormalization(textural_feature_total)
    return textural_feature_total

# 提取 hessian 特征
def extract_hessian_feature(path_image):
    radius = 1;
    n_point = radius * 8;
    textural_feature_total = []
    for files in os.listdir(path_image):
        image = array(Image.open(path_image + files))
        Hrr, Hrc, Hcc = feature.hessian_matrix(image, sigma=0.1, order='rc')
        # 计算hessian特征
        hessian_feature = feature.hessian_matrix_det(image)
        # 统计直方图
        hessian_hist = np.histogram(hessian_feature, bins=256)
        # 每遍历一张图片，将该图片的特征向量添加到list后面
        textural_feature_total.append(hessian_hist[0])
    return textural_feature_total

# n_components:保留下来的特征的个数
# PCA 特征降维
def get_feature_pca(textural_feature_total):
    lbp_feature_array = np.array(textural_feature_total)
    pca = PCA(n_components=39)
    lbp_feature_pca = pca.fit(lbp_feature_array).transform(lbp_feature_array)
    return lbp_feature_pca

# 将glcm特征和lbp特征按照libsvm格式添加到统一 txt文件中
def create_txt(path_positive,path_negative, path_svm, label_p, label_n):
    #提取正样本的glcm特征
    glcm_feature_p = extract_glcm_feature(path_positive)
    #提取负样本的glcm特征
    glcm_feature_n = extract_glcm_feature(path_negative)
    #提取正样本的lbp特征
    lbp_feature_p = extract_lbp_feature(path_positive)
    #提取负样本的lbp特征
    lbp_feature_n = extract_lbp_feature(path_negative)
    #对lbp特征进行PCA降维
    lbp_pca_p = get_feature_pca(lbp_feature_p)
    lbp_pca_n = get_feature_pca(lbp_feature_n)
    f = open(path_svm, 'w+')
    #正样本图片数量
    data_p_length = len(glcm_feature_p);
    #负样本图片数量
    data_n_length = len(glcm_feature_n);
    #代表一张图片的glcm特征是多少维向量
    glcm_length = len(glcm_feature_p[0]);
    # 代表一张图片的lbp特征是多少维向量
    lbp_length = len(lbp_pca_p[0])

    for j in range(0, data_p_length):
        #将每张图片的标签写入txt
        f.write("%i "%label_p)
        #若提取的cnn特征个数为100，则传统特征标记从101开始
        #将glcm特征写入txt
        for i in range(0,glcm_length):
            vec_temp = glcm_feature_p[j]
            newcontext = "%i:"%(i + 101) + "%f"%(vec_temp[i]) + " "
            f.write(newcontext)
        #将lbp特征写入txt
        for i in range(0, lbp_length):
            vec_temp = lbp_pca_p[j]
            newcontext = "%i:"%(i +glcm_length+ 101) + "%f"%(vec_temp[i]) + " "
            f.write(newcontext)
        f.write('\n')
    for j in range(0, data_n_length):
        f.write("%i "%label_n)
        for i in range(0,glcm_length):
            vec_temp = glcm_feature_n[j]
            newcontext = "%i:"%(i + 101) + "%f"%(vec_temp[i]) + " "
            f.write(newcontext)
        for i in range(0, lbp_length):
            vec_temp = lbp_pca_n[j]
            newcontext = "%i:" % (i + glcm_length + 101) + "%f" % (vec_temp[i]) + " "
            f.write(newcontext)
        f.write('\n')

if __name__ == '__main__':
    create_txt(path_positive,path_negative, path_svm, 2, 1)