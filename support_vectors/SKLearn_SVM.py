#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'

from sklearn import svm

X=[[2,0],[1,1],[2,3]]
y=[0,0,1]
clf=svm.SVC(kernel='linear')
clf.fit(X,y)

print(clf)  #输出SVM参数设置
print('clf.support_vectors_:',clf.support_vectors_) #输出支持向量
print('clf.support_:',clf.support_) #输出支持向量的下标（哪个点是支持向量）
print('clf.n_support_:',clf.n_support_) #每个类别分别找到几个支持向量

predicted=clf.predict([[2,0]])
print(predicted)