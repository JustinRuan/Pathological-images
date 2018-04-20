# 所有路径存储点

"""
   1. 最高分辨率图像使用scale = 20进行采集，这时每个Patch边长为256
   2. 全图使用scale = 2.5进行采集，对应的Patch的边长为32
"""
# googleNet
EXTRACT_SCALE = 20
PATCH_SIZE_HIGH = 256
PATCH_SIZE_LOW = 16

# lenet
#EXTRACT_SCALE = 8
#PATCH_SIZE_HIGH = 32
#PATCH_SIZE_LOW = 4

GLOBAL_SCALE = PATCH_SIZE_LOW / PATCH_SIZE_HIGH * EXTRACT_SCALE  # when lenet , = 1, when googleNet， = 1.25
AMPLIFICATION_SCALE = PATCH_SIZE_HIGH / PATCH_SIZE_LOW  # when lenet , = 8, when googleNet， = 16

EXTRACT_PATCH_DIST = 4
CLASSIFY_PATCH_DIST = 8

#切块生成的正样本存储路径
PATCH_PATH_CANCER = "E:/hesimin/mult_mag/testgrade/grade3/testextract/"
#切块生成的负样本存储路径
PATCH_PATH_NORMAL = "E:/hesimin/mult_mag/testgrade/grade3/normal/"

#CNN模型路径
MODEL_PATH = 'H:/PYPROJECT/PIS/Pathological-images/CNN/Models/'

#训练样本图片路径
PATH_TRAIN_IMAGE = 'E:/hesimin/grade_cancer/data0116large/train/'
#测试样本图片路径
PATH_TEST_IMAGE = 'E:/hesimin/grade_cancer/data0116large/test/'

#libsvm格式文件路径
PATH_SVM = 'H:/PYPROJECT/PIS/Pathological-images/Traditional_features/txt_data/'

#难样本复制地址
HS_PATH = 'H:/PYPROJECT/Workspace/hdata/train/'

#生成的txt文件存储路径，如diff100、libsvm格式的txt训练/测试文件
DATA_FILE_PATH = 'H:/PYPROJECT/PIS/Pathological-images/TransferLearning_CNN/txt_data/'
