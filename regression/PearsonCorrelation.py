#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

import numpy as np
import math

# 皮尔逊相关系数
'''
1 衡量两个值线性相关强度的量
2 取值范围 [-1, 1]:
正向相关: >0, 负向相关：<0, 无相关性：=0
'''

# 相关度r
'''
相关度r=((x-x均)(y-y均))的求和/(x的方差的求和*y的方差求和)的开平方
'''
def computeCorrelation(X, Y):
    xbar = np.mean(X)   #X的均值
    yBar = np.mean(Y)   #Y的均值
    SSR = 0  # 分子
    varX = 0    #X方差求和
    varY = 0    #Y的方差求和
    for i in range(0, len(X)):
        diffXXBar = X[i] - xbar #x真值减去x均值
        diffYYBar = Y[i] - yBar #y真值减去y均值
        SSR += diffXXBar * diffYYBar
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)  # 分母
    return SSR / SST  # 相关度r


# 求R平方值
'''
定义：决定系数，反应因变量的全部变异能通过回归关系被自变量解释的比例。
描述：如R平方为0.8，则表示回归关系可以解释因变量80%的变异。换句话说，如果我们能控制自变量不变，则因变量的变异程度会减少80%
简单线性回归：R^2=r*r
多元线性回归：R^2=(y拟-y均)^2d的求和/(y-y均)^2的求和
'''
def polyfit(x, y, degree):
    result = {}
    coeffs = np.polyfit(x, y, degree)  # 计算回归的系数,degree表示最高次幂
    result['polynamial'] = coeffs.tolist()  #将前面的计算的系数保存到字典
    p = np.poly1d(coeffs)   #使用系数构造回归函数公式，参数为x
    print(p)
    yhat = p(x) #拟合数据得到的结果y值
    ybar = np.mean(y)
    ssreg = np.sum((yhat - ybar) ** 2)  #分子
    print('ssreg:', str(ssreg))
    sstot = np.sum((y - ybar) ** 2)     #分母
    print('sstot:', sstot)
    result['determination'] = ssreg / sstot     #R^2的结果值
    print('result:', result)
    return result


testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]
print('r:', computeCorrelation(testX, testY))   #求解打印相关度r
print('r**2:', str(computeCorrelation(testX, testY) ** 2))   #相关度r的平方

print('R^2:',polyfit(testX, testY, 1)['determination'])  # R平方值，简单线性回归，degree为1，结果与R^2=r**2
