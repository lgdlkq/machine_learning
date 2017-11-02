#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

from sklearn import neighbors
from sklearn import datasets    #python自带的数据集

knn=neighbors.KNeighborsClassifier()    #KNN分类器

'''
Iris数据包含150条样本记录，分剐取自三种不同的鸢尾属植物setosa、versic010r和virginica的花朵样本，每一
类各50条记录，其中每条记录有4个属性：萼片长度(sepal length)、萼片宽度sepalwidth)、花瓣长度(petal length)和花瓣宽度(petal width)。
这是一个极其简单的域。
'''
iris=datasets.load_iris()   #导入datasets自带的iris数据集（花：鸢尾属植物的数据集）
#print(iris)    #输出该iris数据集的内容

#使用KNN建模
knn.fit(iris.data,iris.target)  #第一个参数属性集，第二个参数类别集

predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])   #使用建立的KNN模型预测分类结果,注意此处预测填入的仍为二位数组
print(predictedLabel)

predictedLabel=knn.predict([[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5]])
print(predictedLabel)


