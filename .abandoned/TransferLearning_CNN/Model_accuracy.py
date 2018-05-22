from svmutil import *
import utils

#（待改，制作测试集时需注意标签）
# 训练文件路径
train_path = utils.DATA_FILE_PATH + 'train_data_scale.txt'
# 测试文件路径，即待分割图片切成小块之后的文件
test_path = utils.DATA_FILE_PATH + 'test_data_scale.txt'
#读入训练文件
y, x = svm_read_problem(train_path)
# 训练svm分类模型
m=svm_train(y,x,'-c 2048.0 -g 0.0009 -b 1')
# 直接读取libsvm格式的文件，返回分类标签([])和数据([[]])
y2,x2 = svm_read_problem(test_path)

#输出单张图片预测情况
length=len(open(test_path,'rU').readlines())
print(length)
k = 0
for i in range(0,length,1):
    p_lable, p_acc, p_val = svm_predict(y2[i:i + 1], x2[i:i + 1], m, options='-b 1')