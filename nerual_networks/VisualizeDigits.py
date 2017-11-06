#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

'''
自带手写数字识别库的数据查看
'''
from sklearn.datasets import load_digits
import pylab as pl

digits=load_digits()
print(digits.data.shape)
pl.gray()
pl.matshow(digits.images[0])
pl.show()