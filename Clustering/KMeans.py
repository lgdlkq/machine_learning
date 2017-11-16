#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

import numpy as np

def kmeans(X,k,maxIt):  #X为数据集，k为类别个数，maxIt为最大迭代次数
    numPoints,numDim=X.shape    #获取数据集的行列值
    dataSet=np.zeros((numPoints,numDim+1))  #多一列用于存储类别
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k)]  #随机选取中心点(从总的行中随机选取k行)
    # centroids=dataSet[0:2,:]
    centroids[:,-1]=range(1,k+1)    #为最后一列的类别赋初值
    iterations=0
    oldXCentroids=None
    while not shouldStop(oldXCentroids,centroids,iterations,maxIt):
        print('iterations:\n',iterations)
        print('dataSet:\n',dataSet)
        print('centroids:\n',centroids)
        oldXCentroids=np.copy(centroids)    #此处使用copy是因为旧的中心点和新的中心点是两个独立的部分，若使用==则指向同一个索引，一者改变两翼这也跟着改变
        iterations+=1
        updateLabels(dataSet,centroids) #重新分类
        centroids=getCentroids(dataSet,k)   #重新计算新的中心点
    return dataSet

#迭代终止判断
def shouldStop(oldXCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldXCentroids,centroids)  #返回旧的中心点和新的中心点是否相等

#每次传入新的中心点对数据集进行分类
def updateLabels(dataSet,centroids):
    numPoints, numDim = dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1],centroids)

#判断该点与各个中心点的最短距离并返回对应的中心点的label
def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1]) #求两点的距离
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
    print('minDist:',minDist)
    return label

def getCentroids(dataSet,k):
    result=np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]    #取出所有label=i的行的除最后一列数据
        result[i-1,:-1]=np.mean(oneCluster,axis=0)  #axis=0表示对行求均值(axis=1表示列)   新中心点的计算方法:对所有的该label的点求均值
        result[i-1,-1]=i    #对新的中心点进行label赋值
    print('result center points:\n',result)
    return result

x1=np.array([1,1])
x2=np.array([2,1])
x3=np.array([4,3])
x4=np.array([5,4])
testX=np.vstack((x1,x2,x3,x4))  #将数据垂直（按照行顺序）堆叠
result=kmeans(testX,2,10)
print('final result:\n',result)


















