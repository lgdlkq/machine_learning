# 机器学习

        编译环境：win10 64位专业版 + python3.5.0及py35相应的包
        IDE工具：pycharm2016.3.2版

## 一、决策树
### 1.使用python程序实现
    csv
### 2.使用sklearn自带的库函数
    DictVectorizer、tree、preprocessing
### 3.执行过程：
    <1>.读取csv文件的数据
    <2>.定义feature List和label List
    <3>.循环取出每一组特征值（使用字典获取，添加到List中）和label
    <4>.实例化一个DictVectorizer()，并调用其fit_transform(featureList).toarray()方法进行转换格式(0,1)
    <5>.实例化preprocessing.LabelBinarizer()并调用其fit_transform(labelLiat)将label转换格式(0,1)
    <6>.实例化clf=tree.DecisionTreeClassifier(criterion='entropy')分类器，使用ID3方法，（默认为"gini"）
    <7>.clf=clf.fit(dummyX,dummyY)构建决策树分类，传入feature和label的向量集
    <8>.使用f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)方法获取分类结果,(参数f为文件名,因前面将原读入的特征进行了数据格式转换，此处需使用feature_names=vec.get_feature_names())
        可使用graphviz命令将上述生成的dot文件转换为可视图 命令格式为：dot -T pdf (.dot文件的位置和文件名) -o (文件名).pdf
    <9>.使用predict()函数对设置的测试数据进行结果预测

## 二、最邻近规则KNN算法
        
### 1.整个包分两部分：
    <1>.SKLearnExample.py使用sklearn自带库函数neighbors、datasets,其中datasets为自带的常用训练数据集，通过datasets.load_iris()加载其iris数据集用作训练。
        注：Iris数据包含150条样本记录，分剐取自三种不同的鸢尾属植物setosa、versic010r和virginica的花朵样本，每一类各50条记录，其中每条记录有4个属性：萼片长度(sepal length)、萼片宽度sepalwidth)、花瓣长度(petal length)和花瓣宽度(petal width)。这是一个极其简单的域。
    <2>.knnImplementation.py不使用sklearn，自定义函数实现KNN算法，使用的库函数包括：csv、random、math、operator（实例中用于对结果进行排序）。
### 2.程序编写过程：
    <1>.编写加载数据和数据分割函数，将iris的数据集文件的150组数据按指定split值分割成训练集和测试集，此处采用split=0.67，即使其尽可能比例为2：1。
    <2>.编写距离计算两实例函数，传入两个实例instance及其维度，计算所有维度的平方和，返回去总和的sqrt值。
    <3>.编写返回最近的K个实例函数，其中K为指定的值，一般为奇数，传入参数为训练集，测试的一个实例以及K值。对每一个训练集包含的实例调用步骤2的函数求其到测试实例的距离并存到一个List中。将该List进行升序排列，取出最前面的K个实例即为离测试实例最近的K个训练集的实例并返回。
    <4>.编写获取分类结果函数，将每个测试点前面获取到的最近的K个实例进行预测判断，根据该K个实例所属类别的个数进行结果预测。
    <5>.编写准确度计算函数，传入参数为测试数据集，预测结果集.
    <6>.主函数调用执行部分。
