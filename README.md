# 机器学习
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
