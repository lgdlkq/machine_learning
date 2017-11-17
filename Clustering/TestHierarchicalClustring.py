#!usr/bin/env python3
# coding=utf-8

__author__ = 'lgd'

'''
落日图片的层级聚类
'''
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from Clustering.HierarchicalClustring import hcluster, getheight, getdepth


def drawdendrogram(clust, imlist, jpeg='sunset.jpg'):
    #定义高度和宽度
    h = getheight(clust) * 20
    w = 1200
    deepth = getdepth(clust)    #深度
    scaling = float(w - 150) / deepth
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))  #画根节点出来的竖直线
    drawnode(draw, clust, 10, int(h / 2), scaling, imlist, img)
    # img.save(jpeg)
    img.show()


def drawnode(draw, clust, x, y, scaling, imlist, img):
    if clust.id < 0:    #判断是否为终节点，不是就画线
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        print('val:',y,h1,h2,top,bottom,clust.distance)
        ll = clust.distance * scaling
        print('ll:',ll)
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))
        drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, imlist, img)
        drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, imlist, img)
    else:
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((20, 20))  #创建缩略图
        ns = nodeim.size
        print(x, y - ns[1] // 2)
        print(x + ns[0])
        img.paste(nodeim, (int(x), int(y - ns[1] // 2), int(x + ns[0]), int(y + ns[1] - ns[1] // 2)))


imlist = []
folderPath = 'sunsets'
for filename in os.listdir(folderPath):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(os.path.join(folderPath, filename))
n = len(imlist)
print(n)

features = np.zeros((n, 3))
#获取图片的RGB三个通道的值
for i in range(n):
    im = np.array(Image.open(imlist[i]))
    R = np.mean(im[:, :, 0].flatten())  #flatten()扁平化处理
    G = np.mean(im[:, :, 1].flatten())
    B = np.mean(im[:, :, 2].flatten())
    features[i] = np.array([R, G, B])

tree = hcluster(features)
drawdendrogram(tree, imlist, jpeg='sunset.jpg')
