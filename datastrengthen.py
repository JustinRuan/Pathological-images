from PIL import Image
import os
import os.path
import random
import utils

data_path1 = r'F:\PYPROJECT\Workspace\data4\train\cancer'
data_path2 = r'F:\PYPROJECT\Workspace\data4\train\normal'
data_path3 = r'F:\PYPROJECT\Workspace\data4\test\cancer'
data_path4 = r'F:\PYPROJECT\Workspace\data4\test\normal'

def dataenrich(data_path):
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            currentPath = os.path.join(parent, filename)

            im = Image.open(currentPath)

            num = int(filename.split('.')[0])

            # 逆时针旋转90度///文件名以2开头
            out1 = im.rotate(90)
            # out1.show()
            newname1 = data_path + '\\' + str(num+10000000) + ".jpg"
            out1.save(newname1)

            # 逆时针旋转180度///文件名以3开头
            out2 = im.rotate(180)
            # out2.show()
            newname2 = data_path + '\\' + str(num+20000000) + ".jpg"
            out2.save(newname2)

            # 逆时针旋转270度///文件名以4开头
            out3 = im.rotate(270)
            # out3.show()
            newname3 = data_path + '\\' + str(num+30000000) + ".jpg"
            out3.save(newname3)


    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            currentPath = os.path.join(parent, filename)
            im = Image.open(currentPath)
            num = int(filename.split('.')[0])

            # 水平翻转///文件名以6,7,8,9开头
            out4 = im.transpose(Image.FLIP_LEFT_RIGHT)
            newname4 = data_path + '\\' + str(num+50000000) + ".jpg"
            out4.save(newname4)



if __name__ == '__main__':
    #dataenrich(data_path1)
    #dataenrich(data_path2)
    dataenrich(data_path3)
    dataenrich(data_path4)