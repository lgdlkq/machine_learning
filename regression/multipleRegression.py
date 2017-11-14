#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

import time
from numpy import genfromtxt    #用于将导入的数据转换为numpy.array()类型
from sklearn import linear_model

start=time.time()
deliverData=genfromtxt('Delivery.csv',delimiter=',')

print('read time:',time.time()-start,'s')
print('deliverData:\n',deliverData)

#分割数据
X=deliverData[:,:-1]
Y=deliverData[:,-1]

print('X:\n',X)
print('Y:\n',Y)

regr=linear_model.LinearRegression()    #初始化模型
regr.fit(X,Y)   #训练模型

print('coefficients:\n',regr.coef_) #输出参数预测值(即b0,b1...)
print('intercept:\n',regr.intercept_)   #截距值

xPred=[[102,6]] #注意：此处是一个二维的矩阵，需要与前面训练分割的数据X的维度一致，属性个数需相同
yPred=regr.predict(xPred)
print('predicted Y:',yPred)




















