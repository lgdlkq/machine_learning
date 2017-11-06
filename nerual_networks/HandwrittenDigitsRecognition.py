#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

'''
手写数字识别
'''

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from nerual_networks.NerualNetworkClass import NeuralNetwork
from sklearn.model_selection import train_test_split   #交叉验证分割数据集,model_selectio对应旧版的cross_validation

digits=load_digits()    #加载手写数字识别的数据集
X=digits.data   #取出特征值
y=digits.target #取出label
#将所有值转换到0-1间
X-=X.min()
X/=X.max()

nn=NeuralNetwork([64,100,10],'logistic')    #输入层与维度相同，输出层与要分的类相同，隐藏层一般要比输入层多些
X_train,X_test,y_train,y_test=train_test_split(X,y) #数据分割
labeis_train=LabelBinarizer().fit_transform(y_train)    #将label转换成矩阵的形式，第几个数字就在第几位上设为1，其余设为0
labeis_test=LabelBinarizer().fit_transform(y_test)
print('start fitting...')
nn.fit(X_train,labeis_train,epochs=3000)
predictions=[]
for i in range(X_test.shape[0]):
    o=nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
























