#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing

allElctronicsData=open(r'AllElectronics.csv','r')
reader=csv.reader(allElctronicsData)
headers = next(reader)  #获取CSV文件的第一列
print(headers)

featureList=[]  #特征指的集合
labelLiat=[]    #类别值的集合

#取出相应的值
for row in reader:
    labelLiat.append(row[len(row)-1])#取出类别值（最后一列）
    rowDict={} #字典
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]  #headers[i]作为键值key，依次添加每一行数据
    featureList.append(rowDict) #将每一行的属性和值相应的存储到特征集合（未进行格式转换）
print(featureList)

#将特征集合转换为特征向量，转换为sklearn库需要的格式（0,1格式）
vec= DictVectorizer()#初始化一个特征向量的转换函数
dummyX=vec.fit_transform(featureList).toarray()  #转化
print('dummX:'+str(dummyX))#输出转换后的特征向量
print(vec.get_feature_names())  #输出每个属性的取值
print('labelList:'+str(labelLiat))

#将label分类结果转换为0,1格式
lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labelLiat)
print('dummyY:'+str(dummyY))

#使用决策树分类器分类
clf=tree.DecisionTreeClassifier(criterion='entropy') #使用ID3的分类方法(信息熵增益)
clf=clf.fit(dummyX,dummyY) #分类构建决策树，参数依次为特征向量集，label结果向量集
print('clf:'+str(clf))#输出决策树的相关参数

#将结果可视化
with open('allElectronicInformationGainOri.dot','w') as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
#可以使用graphviz命令将上述生成的dot文件转换为可视图 命令为：dot -T pdf (.dot文件的位置和文件名) -o (文件名).pdf

oneRowX=dummyX[0,:]
print('oneRowX:'+str(oneRowX))

newRowX=oneRowX
newRowX[0] = 1
newRowX[2] = 0
print('newRowX:'+str(newRowX))

predictedY=clf.predict([newRowX])   #此处的newRowX因为为1维数组，需转换为二维数组
print('predictedY:'+str(predictedY))





