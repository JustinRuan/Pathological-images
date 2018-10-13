import os

os.chdir('F:\Anaconda\Anaconda3\Lib\site-packages\libsvm\windows')
from svmutil import *
from grid import *
import utils
train_path = utils.PATH_SVM + 'train_data0116large_merge.txt'
test_path = utils.PATH_SVM + 'test_data0116large_merge.txt'
#读入训练文件
y, x = svm_read_problem(train_path)

m=svm_train(y,x,'-c 2.0 -g 0.03125')

y2,x2 = svm_read_problem(test_path)

p_lable, p_acc, p_val = svm_predict(y2, x2, m)