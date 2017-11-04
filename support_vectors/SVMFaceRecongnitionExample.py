#!usr/bin/env python3
#coding=utf-8

__author__ = 'lgd'
'''
线性不可分SVM
'''
# from __future__ import print_function #用做在py2.x的版本测试py3.x的特性时使用，此处编译环境为py3.5.0

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split   #用来分割数据集的，将数据分为训练数据和测试数据
from sklearn.datasets import fetch_lfw_people   #人脸数据集
from sklearn.grid_search import GridSearchCV    #用来寻找合适的SVM参数组合
from sklearn.metrics import classification_report   #用来给模型打分
from sklearn.metrics import confusion_matrix    #用来给模型打分
from sklearn.decomposition import RandomizedPCA #用于降维
from sklearn.svm import SVC

print(__doc__)

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')    #显示程序进度记录到标准输出

lfw_people=fetch_lfw_people(min_faces_per_person=70,resize=0.4,data_home=r"F:\PythonTrainFaile\scikit_learn_data",download_if_missing=True) #加载数据集，找不到是联网下载，默认位置为C:\user\
n_samples,h,w=lfw_people.images.shape   #数据集的实例个数（图片数），h，w

X=lfw_people.data   #数据集的特征向量的矩阵， 所有的训练数据,1288张图片，每张图片1850个特征值
n_features=X.shape[1]   #获取数据集的特征向量的维度

y=lfw_people.target #数据集每一个实例对应的类别标记
target_names=lfw_people.target_names    #所有类别里面有那些类别
n_classes=target_names.shape[0] #数据集有多少实例区分，几个人需要识别

print('Total dataset size:')
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25)  #数据集分割

#PCA降维
n_components=150    #降维的参数,组成元素的数量，即保留下来的特征个数
print("Extracting the top %d eigenfaces from %d faces" % (n_components,x_train.shape[0]))
t0=time()
pca=RandomizedPCA(n_components=n_components,whiten=True).fit(x_train)   #使用随机的PCA降维方法建模
print("done in %0.3fs" % (time()-t0))

eigenfaces=pca.components_.reshape((n_components,h,w))  #提取人脸特征

print('projecting the input data on the eigenfaces orthonormal bosis')
t0=time()
x_train_pca=pca.transform(x_train)  #训练集PCA降维,降到150维
x_test_pca=pca.transform(x_test)    #测试集PCA降维，降到150维
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0=time()
param_grid={'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],}    #定义多种核函数组合，以便在下面选择准确率高的组合作为参数，C 是对错误部分的惩罚；gamma 合成点
clf=GridSearchCV(SVC(kernel='rbf'),param_grid)  ##rbf处理图像较好，C和gamma组合，穷举出最好的一个组合
clf=clf.fit(x_train_pca,y_train)    #SVM建模
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)  #输出最好的模型的信息

print("Predicting people's names on the test set")
t0 = time()
y_pred=clf.predict(x_test_pca)
print("done in %0.3fs" % (time() - t0))

#对于下述打印出的结果：precision是预测的准确率，recall是召回率f1-score是一个兼顾考虑了Precision和Recall的评估指标。他们的数值越接近1说明预测的越准
print(classification_report(y_test,y_pred,target_names=target_names))   #输出预测结果准确率
#confusion_matrix混淆矩阵验证:如果全部都是100%预测，那么数据应该都排列在对角线上，也就是说，每一个行列对应之后就会在对角线上+1
print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))

def plot_gallary(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.90,hspace=0.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape(h,w),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred,y_test,target_names,i):
    pred_name=target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name=target_names[y_test[i]].rsplit(' ',1)[-1]
    return 'predicted: %s\ntrue:     %s' % (pred_name,true_name)
#把预测出来的人名存起来
prediction_titles=[title(y_pred,y_test,target_names,i)for i in range(y_pred.shape[0])]
plot_gallary(x_test,prediction_titles,h,w)
#提取过特征向量之后的脸是什么样子
eigenface_titles=["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallary(eigenfaces,eigenface_titles,h,w)

plt.show()




