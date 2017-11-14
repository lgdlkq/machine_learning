#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

import numpy as np

'''
线性回归的函数为：y=b0+b1*x
b1=(Xi-x)(Yi-y)的累计求和/(Xi-x)**2的累计求和  其中x为X集的均值，y为Y集的均值,最小二乘法的变形（重点）
b0=y-b1*x
'''
def fitSLR(x,y):
    n=len(x)
    dinominator=0   #分母
    numerator=0 #分子
    for i in range(0,n):
        numerator+=(x[i]-np.mean(x))*(y[i]-np.mean(y))  #numpy.mean(x)可以求出x的均值
        dinominator+=(x[i]-np.mean(x))**2

    print('numerator:',numerator)
    print('dinominator:',dinominator)

    b1=numerator/float(dinominator)
    b0 = np.mean(y) - b1 * np.mean(x)

    return b0,b1

# def ercheng(x,y):
#     n=len(x)
#     x=np.array(x)
#     y=np.array(y)
#     dinominator = 0  # 分母
#     numerator = 0  # 分子
#     dinominator=n*x.dot(y) - sum(x)*sum(y)
#     numerator=n*x.dot(x) - sum(x)**2
#     b1=dinominator/float(numerator)
#     b0=np.mean(y)-b1*np.mean(x)
#     return b0,b1

def predict(x,b0,b1):
    return b0+b1*x



x=[1,3,2,1,3,4,5]
y=[14,24,18,17,27,29,32]

b0,b1=fitSLR(x,y)
print('intercept:',b0,' slope:',b1)

x_test=2
y_test=predict(x_test,b0,b1)
print('y_test:',y_test)




