#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

import numpy as np
import pylab as pl  # 画图包
from sklearn import svm

# 创建四十个点
# np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
print('X:',X)
print('Y:',Y)
# 建立SVM模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

predicted=clf.predict([[2,2]])
print(predicted)

print(clf)  #输出SVM参数设置
print('clf.support_vectors_:',clf.support_vectors_) #输出支持向量
print('clf.support_:',clf.support_) #输出支持向量的下标（哪个点是支持向量）
print('clf.n_support_:',clf.n_support_) #每个类别分别找到几个支持向量

# 公式：w_0x + w_1y +w_3=0，变形为：y = -(w_0/w_1) x + (w_3/w_1)
# 超平面的分割直线和其上下的边界线的直线函数
w = clf.coef_[0]  # 模型的W值
a = -w[0] / w[1]  # 表示超平面分割直线的斜率
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]  # 找到的超平面分割直线，而clf.intercept_[0]表示公式中的W[3]

b = clf.support_vectors_[0]  # 第一个支持向量
yy_down = a * xx + (b[1] - a * b[0])  # 超平面分割线下面的那条边界线
b = clf.support_vectors_[-1]  # 最后一个支持向量
yy_up = a * xx + (b[1] - a * b[0])  # 超平面分割线上面的那条边界线

# print('w:',w,' a:',a)
# print('yy:',yy)
# print('yy_down:',yy_down)
# print('yy_up:',yy_up)

# 绘图显示
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='red')
pl.scatter(X[:, 0], X[:,1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
