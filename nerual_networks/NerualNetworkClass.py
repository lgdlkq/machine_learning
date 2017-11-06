#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

'''
简单的神经网络算法
'''

import numpy as np


def tanh(x):  # 非线性转换方程，双曲线函数  tanhx=(e^x-e^(-x))/(e^x+e^(-x))
    return np.tanh(x)


def tanh_deriv(x):  # 双曲函数的导数
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):  # 非线性转换方程，logistic(逻辑)函数
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):  # logistic(逻辑)函数的导数
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork(object):
    def __init__(self, layers, activation='tanh'):
        """
        self:相当于JAVA的this
        :param layers:相当于Java的List，神经网络有几层，每层有多少神经元（几个值就有几层，每个值代表每层有多少神经元）
        :param activation:模式，用户选取神经网络可以选择双曲函数还是logistic函数，默认为双曲函数
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []  # 存放神经网络的神经元之间的weight
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        '''
        :param X: 数据集
        :param y: label标签
        :param learning_rate:学习率，通常设在0~1之间，梯度算法中的步长
        :param epochs: 抽样更新的次数（循环次数）
        :return:
        '''
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])  # 初始化一个全为一的矩阵,列数比X多一，表示对偏向bias的初始化
        temp[:, 0:-1] = X  # 第一个:表示取所有的行，第二个0:-1表示列数取第一列到除了最后一列,向输入层添加偏置单元
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 随机抽取一行
            a = [X[i]]
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l]))) #对前面取出的数据集的每一个属性值进行权重点积，a开始时只有输入层的输入实例，结果：开始求出隐藏层，最后求出输出层
            error = y[i] - a[-1]    #真实值与预测值的误差
            deltas = [error * self.activation_deriv(a[-1])] #输出层的误差

            for l in range(len(a) - 2, 0, -1):  #从最后一层回退到第零层，每次回退一层，参数依次对应，开始回退的层数，回退结束的层数，每次回退的层数
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))  #添加回退的隐藏层的误差

            deltas.reverse()    #因为是回退计算，所以讲deltas的元素前后颠倒
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)   #权重更新

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
