# 绘制loss图
# 1.train loss不断下降，test loss也不断下降： 网络在认真学习

# 2.train loss不断下降，test loss趋于不变： 网络过拟合

# 3.train loss趋于不变，test loss趋于不变： 学习遇到瓶颈，需减小lr或batch_size

# 4.train loss趋于不变，test loss不断下降： 数据集有问题

# 5.train loss不断上升，test loss不断上升（最终变为NaN）：网络结构不当，训练超参设置不当等

import linecache
import matplotlib.pyplot as plt

#载入网络训练日志
f = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/log.txt")

#提取训练时loss和lr
f2 = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/loss.txt", "w+")
f3 = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/lr.txt", "w+")

for lines in f.readlines():
    if(lines.find("), loss = ") > 0):
        loss = lines.split("), loss = ")[-1]
        f2.write(loss)
f2.close()

#????
f.close()


f = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/log.txt")



for lines in f.readlines():
    if(lines.find("lr = ") > 0):
        lr = lines.split("lr = ")[-1]
        f3.write(lr)
f3.close()



f4 = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/loss.txt" , "rU")
count = len(f4.readlines())

f5 = open("F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/lr.txt" , "rU")
count2 = len(f5.readlines())

print(count)
print(count2)

# #train_loss坐标
x = []
y = []

# #test_loss坐标，暂时手动记录,一个epoch一次测试
x2 = [0,1800,3600,5400,7200,9000,10800,12600]
y2 = [6.84482,0.26554,0.0600643,0.0600349,0.11,0.055,0.058,0.055]

for i in range(0, count):
    X = i*100
    Y = linecache.getline(r'F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/loss.txt',i+1)

    x.append(X)
    y.append(Y)

    i += 1


#lr坐标
x3 = []
y3 = []

for j in range(0, count2):
    X = j * 100
    Y = linecache.getline(r'F:/PYPROJECT/Workspace/Pathological-images/DetectCancer/lr.txt',j+1)

    x3.append(X)
    y3.append(Y)

    j += 1


plt.figure(figsize = (8,4))
plt.plot(x, y, color = "blue", linewidth = 1, label = "train_loss")
plt.plot(x2, y2,color = "red", linewidth = 1, label = "test_loss")
plt.legend(loc = 'upper right')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("Train&Test Loss Curve")
plt.show()
#plt.savefig("Train&Test Loss Curve.jpg")

plt.figure(figsize = (8,4))
plt.plot(x3, y3, color = "blue", linewidth = 1, label = "lr")
plt.xlabel("iterations")
plt.ylabel("lr")
plt.title("Learning Rate")
plt.show()
#plt.savefig("lr.jpg")