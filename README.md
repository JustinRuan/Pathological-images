# 项目共分5个模块
1. 数据预处理 Data_Preprocessing
2. 卷积神经网络 CNN
3. 基于迁移学习的卷积神经网络 TransferLearning_CNN
4. 传统特征抽取 Traditional_features
5. 大幅图像区域分割 Svm_seg.py

## 一、数据预处理 Data_preprocessing

### 1. 切片文件：Test_patch.py
将WSI文件在固定的放大倍数下进行样本切块

### 2. 生成样本路径作为卷积神经网络训练的数据接口：Creat_txt.py
给正负样本标签和文件名，写入TXT文件

### 3. 数据增强：Data_strengthen.py
当数据集稀缺时，通过旋转、缩放、平移、翻转等操作可将数据量扩增为原始数据集的八倍

## 二、卷积神经网络 CNN

### 1. 训练模型 Models
内含两个网络结构：AlexNet和GoogLeNet的配置文件
* (1)solver.prototxt：网络训练参数
* (2)train_val：待训练的网络结构
* (3)deploy：拿来完成分类任务的网络结构

### 2.网络的训练：train.py
加载网络模型和训练参数并开始模型的训练

### 3.模型准确率测试：Model_accuracy.py
模型训练完后，加载模型，将测试集输入到分类deploy中，输出分类器的样本类别概率值，计算准确率
同时将分类模糊(正负类别概率值之差小于0.3)的样本存储到难样本文件夹HS_data以备后续研究

## 三、基于迁移学习的卷积神经网络 TransferLearning_CNN

该模块运用具有泛化能力的卷积神经网络模型作为特征抽取模型，提取正负样本的特征并找到其中差异最大的100个特征的维度
所使用的模型为'CNN/Models/Accurate_model/bvlc_reference_caffenet.caffemodel'

### 1.获得特征值差异最大的100个维度的索引：Diff100.py
特征抽取模型抽取正负样本特征后，对应维度特征相减，找到差值最大的维度，即正负样本差异性最大的特征，记录索引，写入diff100.txt

### 2.制作SVM训练集：Train_data.py
将训练数据集输入特征抽取网络，依照diff100.txt制作SVM训练集，训练集中单张样本包含1个标签和100个特征值，写入train_data.txt

### 3.制作SVM测试集：Test_data.py
在大幅WSI图像中采用滑动窗口滑动切割图像，将切割得到的样本输入到特征抽取模型中，同样按照diff100.txt制作SVM测试集，写入seg_data.txt


## 四、传统特征抽取 Traditional_features

### 1.提取传统特征：Extract_feature.py
先提取glcm特征，lbp特征（PCA降维），再将glcm特征和lbp特征拼接写在同一txt文件中（归一化）由于之前提取的CNN特征维度为100，这里
的传统特征索引从101开始，便于后续特征拼接

### 2.合并CNN特征和传统特征：Merge_feature.py
分别读取存有cnn特征和传统特征的txt文件，按行拼接，生成一个145维的特征向量集

### 3.分类器Svm_classifier.py
使用合并后的特征集训练SVM模型，训练完后对测试集分类，输出准确率。

## 五、大幅图像的癌细胞区域分割 Svm_seg.py

归一化train_data.txt和seg_data.txt，使用train_data.txt训练SVM模型，训练前先用grid.py求训练最优参数(c,g)值。
训练完后对seg_data.txt进行分类，将分类结果映射到WSI图像中，其中若分类结果为正则增强该窗口区域的色度，通过色度叠加凸显癌细胞区域
