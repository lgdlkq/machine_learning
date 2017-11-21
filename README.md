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

## 三、支持向量机SVM
### 1.使用类库包：
    numpy、pylab、sklearn.svm、time、logging、matplotlib.pyplot、sklearn.cross_validation.train_test_split、sklearn.datasets.fetch_lfw_people、sklearn.grid_search.GridSearchCV、sklearn.metrics.classification_report、sklearn.metrics.confusion_matrix、
    sklearn.decomposition.RandomizedPCA
    说明：sklearn.cross_validation.train_test_split、sklearn.grid_search.GridSearchCV在Python3.5中已迁移至sklearn.model_selection.train_test_split、sklearn.model_selection.GridSearchCV
### 2.三个程序：
    <1>.SKLearn_SVM.py,简单的SVM测试
    <2>.SKLearnSVMExample.py,使用随机数创建四十组数据，前20个做训练集，后20个做测试集
    <3>.SVMFaceRecongnitionExample.py,使用训练集做人脸检测的训练分类
    说明：前两个程序为线性可分的SVM，后者为线性不可分SVM
### 3.程序过程：
    (1).线性可分SVM，第二个程序：
        <1>.使用numpy库的random.randn函数生成四十组数据
        <2>.建立SVM模型，设置核函数为'linear'并进行fit训练，尝试对某个数据预测并输出预测结果
        <3>.输出该SVM的相关信息
        <4>.取出SVM相关的W、h值对公式w_0x + w_1y +w_3=0，
        变形为：y = -(w_0/w_1) x + (w_3/w_1)构建该函数（超平面的分割线）与其两条边界线函数
        <5>.使用numpy的linspace函数构建一组X值，通过上述构建的函数运用绘图函数将结果绘制在图上
    (2).线性不可分SVM，第三个程序：
        <1>.载入数据集，fetch_lfw_people默认将自动去user目录下查找相应的数据集(如果使用data_home指定了数据集的位置将去相应位置查找)，如果未找到将自动联网下载（若指定参数download_if_missing=False则不会下载，下载文件较大比较耗时，数据集大小为220MB左右）
        <2>.载入数据后，取出数据集的实例个数、h、w、特征向量矩阵及其维度、每一个实例的类别标记、所有的类别种类、需检测区分的实例个数等数据
        <3>.数据集分割
        <4>.设置PCA降维参数，使用随机的PCA降维方法建立PCA降维模型
        <5>.分别对训练集和测试集进行PCA降维
        <6>.定义多种核函数参数的组合，包括错误惩罚C和gamma
        <7>.使用GridSearchCV函数构建SVM模型，参数指定：核函数为rbf，以及传入前面定义的核函数组合，GridSearchCV函数会自动寻找组合中的最佳组合参数
        <8>.训练SVM模型并使用该训练好的SVM模型对测试集的实例进行预测分类
        <9>.使用classification_report输出预测结果的准确率，输出的结果显示属性：precision是预测的准确率，recall是召回率f1-score是一个兼顾考虑了Precision和Recall的评估指标。他们的数值越接近1说明预测的越准
        <10>.使用confusion_matrix输出混淆矩阵验证，若正确率100%，数据将都排列在对角线上，不在对角线上的数据越多准确率越差
        <11>.定义绘图板函数，预测结果对应的分类名称提取函数
        <12>.调用上述定义的函数把预测的分类名称取出，将预测结果和实际结果与相应的图像绘制在第一个绘图板
        <13>.提取特征并将提取特征后的结果绘制在第二个绘图板
        <14>.将绘图结果显示出来

## 四、NeuralNetwork
### 1.使用类库包：
    numpy、sklearn.datasets.load_digits、sklearn.metrics.confusion_matrix,classification_report、sklearn.preprocessing.abelBinarizer、nerual_networks.NerualNetworkClass.NeuralNetwork、sklearn.model_selection.train_test_split
### 2.四个.py文件：
    <1>.NerualNetworkClass.py定义NeuralNetwork类，其余.py文件进行训练和预测时使用；该类是对NeuralNetwork算法的实现，采用了交叉验证的方法加快训练（多次随机抽取数据集的一行训练，而非对每行都依次训练），同时，在终止训练的条件选择上，为了简便只选取了限定交叉验证的训练次数，未添加通过误差等判断的条件
    <2>.NNSimpleTest.py是简单的NN算法的测试，调用了NerualNetworkClass.py中定义的NerualNetwork的类
    <3>.HandwrittenDigitsRecognition.py是手写数字识别的简单例子，同样调用了NerualNetworkClass.py中定义的NerualNetwork的类
    <4>.VisualizeDigits.py查看sklearn自带手写数字识别库的数据
### 3.代码实现：
    <1>.重点：NerualNetworkClass.py中NerualNetwork类的实现：
        （1）.首先，定义两种（偏向）S函数：双曲函数tanh(x)、logistic(逻辑)函数logistic(x)以及两种函数的导数函数tanh_deriv(x)、logistic_derivative(x)
        （2）.NeuralNetwork(object)的定义：1.定义构造函数，传入参数为神经网络的层数及每层的节点个数（使用List传入，其长度即为层数）、采用的偏向函数，通过判断采用的函数类型为类的属性进行方法的选择，同时使用随机数的方式为神经网络进行权重的初始化
        （3）.对NeuralNetwork类定义fit训练函数，参数为：数据集，label集，学习率，需要进行训练更新的次数epochs。先将数据集X转换为array型以便计算，同时初始化一个全为1的矩阵，大小为X的行*（X的列+1），多的一列用于初始化偏向bias为1，然后将前面的部分设为X的值并将整个矩阵赋给X，再将label集y转换为array
        （4）.fit中编写更新操作：随机取出一行数据，对改行数据进行相应的加权求内积并存储（开始只有输入层的实例，最后求出输出层），求出输出层的真实值与预测值误差，再利用其求出输出层的偏向误差；从最后一层回退到第零层，每次回退一层，循环求出各个偏向值并存取；因为是回退计算，需要将保存偏向元素进行前后颠倒，然后进行循环更新权重。将此步骤进行循环，循环次数为传入的epochs。
        （5）.对NeuralNetwork类定义predict预测函数,数据处理同步骤三（不需要对label处理），求其预测值并返回。
    <2>.HandwrittenDigitsRecognition.py的实现：
        （1）.加载手写数字识别的数据集，取出特征值和label值，再将所有值转换到0-1间
        （2）.初始化NeuralNetwork
        （3）.对步骤一载入的数据进行分割
        （4）.将label转换成矩阵的形式，第几个数字就在第几位上设为1，其余设为0（使用LabelBinarizer().fit_transform方法）
        （5）.开始训练
        （6）.对分割出的测试集进行预测并将预测结果存取，输出预测结果

## 五、regression
### 1.使用的类库：
    numpy、sklearn.linear_model、numpy.genfromtxt
### 2.四个.py文件两个.csv文件
    <1>.SimpleLinearRegression.py:线性回归实例，使用最小二乘法定义回归函数
    <2>.multipleRegression.py、multipleRegressionNew.py：前者使用Delivery.csv文件的数据，后者使用Delivery_Dummy.csv的数据，两者的过程并无区别，主要区别在于后者是前者的数据衍化，前者为值预测，后者演化为分类问题
    <3>.PearsonCorrelation.py计算皮尔顿相关系数（相关度r、R平方值(决定系数)）
### 3.代码实现：
    <1>.SimpleLinearRegression.py:使用最小二乘法，故定义最小二乘法的函数，定义预测函数，使用数据训练，对测试数据使用训练的结果进行预测
    <2>.multipleRegression.py、multipleRegressionNew.py：读入.csv的文件数据，使用sklearn.linear_model库进行数据训练，对预测数据进行预测
    <3>.PearsonCorrelation.py：相关度r=((x-x均)(y-y均))的求和/(x的方差的求和*y的方差求和)的开平方、简单线性回归：R^2=r*r、多元线性回归：R^2=(y拟-y均)^2d的求和/(y-y均)^2的求和

## 聚类Clustering
### 1.使用的类库：
    numpy
    PIL.Iamge、PIL.ImageDraw
### 2.三个.py文件
    1.KMeans.py：无监督学习的简单聚类算法K-means的一个实例
    2.HierarchicalClustring.py：层级聚类的算法实现
    3.TestHierarchicalClustring.py：层级聚类的一个实例，使用了2定义的方法
### 3.算法基本思想：
     选择K个点作为初始质心，将每个点指派到最近的质心，形成K个簇， 重新计算每个簇的质心，直到簇不发生变化或达到最大迭代次数时完成分类，其中K代表了分类的类别个数，可根据实际需要指定
     注：该算法不稳定，不同的初始质心可能得到不同的分类结果，适合在没有明确分类的情况且对稳定性要求不高情况下使用，算法实现简单
### 4.代码实现：
    1.KMeans.py执行过程：
        <1>.载入数据集
        <2>.将数据集转换为numpy的类型并为每一组数据初始化分类（此处为由1到K依次指定）
        <3>.循环判断是否收敛或者达到设定的迭代次数
        循环内部：
        <1>.将现在的质心赋给旧质心存储变量
        <2>.在现在的分类基础和质心条件下进行重分类
        <3>.重新计算新的质心
        <4>.循环上述过程
    2.HierarchicalClustring.py定义：
        <1>.定义cluster_node类，用来存放算法及执行过程中的节点（包括其数据、左节点、右节点、距离、判断是否去玩的标志等）
        <2>.定义两种不同的距离算法：sqrt(sum((v1 - v2) ** 2))和sum(abs(v1 - v2))，v1,v2分别为两个点的坐标向量
        <3>.定义层级聚类的实现方法（结果是类似树形结构），将每一行数据都作为一个类别初始化，循环判断最近距离（默认为sqrt计算），并用一个新的节点向量代最近的两个节点的向量，将这两个节点分别作为其左右节点，类别减一，直到只剩一个类别，最后返回聚类后的最后一个节点（所有的数据节点都可以通过这个根节点从其左右节点开始遍历得到）
        <4>.定义方法实现取出建好的“树”的任意高度的分类情况（使用递归遍历左右节点并把遍历的结果依次返回累计作为最后的返回结果）
        <5>.定义方法实现取出生成的熟悉部分结构的聚类元素（每一个子类），同样采用递归遍历左右节点并返回累计结果得到
        <6>.定义方法实现打印出聚类的给个节点，同样递归遍历左右节点输出
        <7>.定义实现获取聚类结果“树”的高度和深度的方法
    3.TestHierarchicalClustring.py的测试：
        <1>.对100张落日图片进行RGB三个通道像素的扁平化处理，并将每一张图片得到的数据作为一行数据，最后构成一个数据集
        <2>.调用HierarchicalClustring.py中3步骤定义的方法训练得到层级聚类的结构（结果）
        <3>.定义函数画出该层级聚类后的结构图，最末端的叶子节点使用每张对应节点的图的缩略图显示
        <4>.调用3步骤定义的方法绘出1步骤得到的层级聚类结构图