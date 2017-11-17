#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

from numpy import *

'''
层级聚类的算法实现
'''

# HierarchicalClustring算法的过程形似一颗树形结构,节点定义类
class cluster_node(object):
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        '''
        :param vec: 数据集的一行向量
        :param left: 左节点
        :param right: 右节点
        :param distance: 两点距离
        :param id: 用于取出生成树形结构的聚类的元素时判断元素是否取完
        '''
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count


# 两种不同的距离算法
def L2dist(v1, v2):
    return sqrt(sum((v1 - v2) ** 2))


def L1dist(v1, v2):
    return sum(abs(v1 - v2))


def hcluster(features, distance=L2dist):
    distances = {}  #使用字典存储计算得到的距离值
    currentclustid = -1  # f当前的跟踪的类的id
    clust = [cluster_node(array(features[i]), id=i) for i in range(len(features))]  # 把每一行的数据作为一个实例类别，id依次递增
    while len(clust) > 1:  # 当类别个数大于1,直到只有一个类别是停止
        lowestpair = (0, 1) #每次要选的相似度最高的类别组
        closest = distance(clust[0].vec, clust[1].vec)  #初始化一个最近距离
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)    #把两个类之间的距离存储到字典中
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest: #判断是否是最短距离
                    closest = d
                    lowestpair = (i, j) #最近两类的下标
        mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 for i in range(len(clust[0].vec))]    #两类的平均距离求出的是新类别的数据
        newcluster = cluster_node(array(mergevec), left=clust[lowestpair[0]], right=clust[lowestpair[1]],
                                  distance=closest, id=currentclustid)  #用一个新的节点替代上面最近的两个节点（类别）进行下次迭代的类别
        currentclustid -= 1
        #删掉归为一类的两个类
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        #将两个类归类后的替代类添加到类集合中
        clust.append(newcluster)
    return clust[0] #返回最后的只剩一个类时的结果（根节点）

#取出建立好的树形结构的任意高度（截取到需要的类的个数时的分类情况）
def extract_cluster(clust, dist):
    '''
    :param clust: 生成的树
    :param dist: 截取的高度（从下开始计数的高度）
    :return:
    '''
    clusters = {}
    if clust.distance < dist:
        return [clust]
    else:
        cl = [] #左节点
        cr = [] #右节点
        if clust.left != None:
            cl = extract_cluster(clust.left, dist=dist)
        if clust.right != None:
            cr = extract_cluster(clust.right, dist=dist)
        return cl + cr

#取出生成的树形结构的聚类的元素（每一个子类）
def get_cluster_elements(clust):
    if clust.id >= 0:   #hcluster(features, distance=L2dist)函数将id都设为了小于等于0
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)
        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        return cl + cr

#打印出树形结构的节点
def printtclass(clust, labels=None, n=0):
    for i in range(n):
        print(' ')
    if clust.id < 0:    #表示当前的节点是一个支点
        print('-')
    else:
        if labels == None:
            print('id:',clust.id)
        else:
            print('labels:',labels[clust.id])
    if clust.left != None:
        printtclass(clust.left, labels=labels, n=n + 1)
    if clust.right != None:
        printtclass(clust.right, labels=labels, n=n + 1)

#获取树的高度
def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    return getheight(clust.left) + getheight(clust.right)

#获取树的深度
def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance
