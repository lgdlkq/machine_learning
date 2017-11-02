#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

'''
KNN算法的实现过程和测试
'''

import csv
import random
import math
import operator

#加载数据，分为训练集和测试集
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    '''
    :param filename: 文件名
    :param split:分割界限值
    :param trainingSet: 分割得到的训练集
    :param testSet: 分割得到的测试集
    :return:
    '''
    with open(filename,'r') as csvfile:
        lines=csv.reader(csvfile)   #读取所有行的数据
        dataset=list(lines) #转化为list格式
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])  #将读取到的每一个数据转换为float类型，以便使用
            if random.random() < split:#随机生成数满足分割界限的要求
                trainingSet.append(dataset[x])  #将改行数据加入到训练集
            else:
                testSet.append(dataset[x])   #将改行数据加入到测试集

#计算距离
def euclideanDistance(instance1,instance2,length):
    '''
    :param instance1:实例1
    :param intance2:实例2
    :param length: 维度大小
    :return:距离
    '''
    distance=0
    for x in range(length):
        distance+= pow((instance1[x]-instance2[x]),2)   #计算所有维度的平方和
    return math.sqrt(distance)

#返回最近的K个实例
def getNeighbors(trainingSet,testInstance,k):
    '''
    :param trainingSet: 训练集
    :param test_instance: 测试集的一个实例
    :param k: 需要的最近的实例的个数
    :return:最近的K个实例
    '''
    distances=[]
    length=len(testInstance)-1#维度大小
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)  #计算测试实例到训练集的每一个实例的距离
        distances.append((trainingSet[x],dist)) #将每一个实例的计算结果添加到distances中
    distances.sort(key=operator.itemgetter(1))  #对计算的结果进行升序排列
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])   #取出最近的K个实例
    return neighbors    #返回最近的K个实例

#分类结果
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True) #分类结果的个数大小降序排列,key=operator.itemgetter(1)指定按第一列值的大小排列（此处为分类结果个数），reverse=True将dict的key与value反向
    return sortedVotes[0][0]   #返回分类结果

#预测准确度
def getAccuracy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))*100.0


def main():
    trainingSet=[]
    testSet=[]
    split=0.67  #划分训练集和测试集的比例为2:1
    loadDataset(r'irisdata.txt',split,trainingSet,testSet)
    print('Train Set:'+repr(len(trainingSet)))
    print('Test Set:'+repr(len(testSet)))

    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        result=getResponse(neighbors)
        predictions.append(result)
        print('> predicted='+repr(result)+', actual='+repr(testSet[x][-1]))
    accuracy=getAccuracy(testSet,predictions)
    print('Accuracy: '+ repr(accuracy)+'%')

if __name__ == '__main__':
    main()

