import utils

#将传统特征和CNN特征合并后的libsvm格式txt文件
path_svm = utils.PATH_SVM + 'train_data0116large_merge.txt'
#合并后的用于svm测试的txt文件
path_test_svm = utils.PATH_SVM + 'test_data0116large_merge.txt'
# 待合并的CNN特征训练文件路径
path_feature_alexnet_train = utils.DATA_FILE_PATH + 'train_data0116large_scale.txt'
# 待合并的CNN特征测试文件路径
path_feature_alexnet_test = utils.DATA_FILE_PATH + 'test_data0116large_scale.txt'
# 待合并的传统特征训练文件路径
path_train_tra = utils.PATH_SVM + 'train_data0116large_lbp_glcm_scale.txt'
# 待合并的传统特征测试文件路径
path_test_tra = utils.PATH_SVM + 'test_data0116large_lbp_glcm_scale.txt'

feature_cnn =[]
feature_traditional =[]
f=open(path_test_svm,'w')
with open(path_feature_alexnet_test, 'r') as f1:
    for line in f1:
        line=line.strip()
        feature_cnn.append(line)
with open(path_test_tra, 'r') as f2:
    for line2 in f2:
        line2=line2.strip()
        line2=line2[2:]
        feature_traditional.append(line2)
for i in range(0,len(feature_traditional)):
    result=feature_cnn[i]+' '+feature_traditional[i]+'\n'
    f.write(result)