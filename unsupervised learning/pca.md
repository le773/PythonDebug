### 1.0 PCA
Principal Components Analysis:主成分分析
主成分是由数据中具有最大方差的方向决定的，因为可以最大程度的保留信息量。

PCA擅长用于旋转、偏移坐标轴，使之达到降维的作用；
PCA将几个相关的特征用一个特征来表示，即将多维降为一维。；
其实我们不必知道那几个特征相关，我们可以直接把许多特征丢进PCA，PCA会自动帮我们分析哪些特征可以合并降维。 

##### 是否可以舍弃特征？

pca会找出一种方法，来将各种输入特征中的信息合并起来；所以，如果在进行PCA之前排除所有输入特征，某种程度上讲，也就排除PCA能够挽救的信息,(如果有很多不相关的特征，是可以舍弃的，但是要小心)；制作主成分之后对其进行特征选择是可以的。

#### 1.1 PCA的特点
- PCA将输入特征转化为主成分的系统化方式
- 这些主成分可以作为新特征使用
- 主成分的定义是数据中会使方差最大化的方向，将出现信息丢失的可能性降至最低
- 可以对主成分划分等级，数据因特定主成分而产生的方差越大，那么该主成分的级别越高，方差最大的主成分即为第一个主成分
- 主成分在某种意义上是**相互垂直**的，第二个主成分基本不会与与第一个主成分重叠，可以将他们作为单独的特征对待
- 主成分数量有限，最大值等于数据集中的输入特征数量

#### 1.2 什么时候使用PCA
- 想要访问隐藏的特征，这些特征可能隐藏显示在你的数据图案中
- 降维
a. 可视化高维数据
b. 怀疑数据中存在噪声，可以帮助抛弃重要性小的特征，**去除噪声**
c. 让算法在少量的特征上更有效，比如说人脸识别，可以将维度降低1/10，然后使用svm来进行训练，之后发现被拍摄人的身份
d. 防止过拟合
- 提高算法效率、精确度


#### 1.3 选择主成分
###### PCA过拟合
当使用大量 PCA 时，会看到任何过拟合迹象, 性能开始下降。

###### 主要原则
主要原则：必须最大程度的保留原信息的数据。

###### 停止条件
通过增加主成分，观察f1得分(等)，如果出现停滞(下降)则停止增加。


![主成分](http://img.blog.csdn.net/20171206233759666?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhuaW5nMTJM/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
通过PCA还可以得出这条轴的一个散布值 
如果散布率较小，该散布值对于第一条变量轴来说倾向于一个很大的值，而对于第二条变量轴来说则小得多 
所以这个数字，碰巧是一个特征值，结果来自于PCA执行的一种特征值分解；

#### 1.4 轴何时占主导地位

![轴何时占主导地位](http://img.blog.csdn.net/20171206235513061?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhuaW5nMTJM/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
##### 长轴是否占优势？
长轴的重要值，特征值远远大于短轴的特征值

#### 1.5 最大方差与信息损失

![信息损失](https://i.imgur.com/AeYD61A.png)

丢失的信息就是数据中的点到主成分方向的距离，所以我们要寻找的就是总距离最小时主轴的方向，即主成分方向。 
根据数学知识数据最大方差的方向就是总距离最小的方向(实际上将点与其在该线上的投影之间的距离最小化(信息丢失最小))
即主成分方向就是由该数据最大方差的方向。 

![信息损失2](https://i.imgur.com/2kcb8ki.png)
上图中绿色点的损失大于黄色点。

### 2.0 特征选择
#### 2.1 相关性
相关性是指该特征与label的判断是否有帮助，有帮助则是相关的，无帮助就是不相关的。而相关性的强弱可以通过该特征的必要程度来决定。 
###### 例子 特征ABCD 
A+B可以判断label,A+C也可以判断label 
可以说A的相关性强，因为它不可替代；而B和C的相关性弱，因为它们之间可以互相替代。D则是不相关。

#### 2.2 有用性
是指一个特征虽然不相关，但是它在某一个特定算法中会起到比较大的作用，则指这个特征是有用的，具有有用性。


#### 2.3 特征增加

1. 根据自己经验直觉觉得哪个feature可以用来训练
2. code the feature
3. 可视化feature
4. 重复该步骤

#### 2.3 特征删除

尽量减少特征，简化学习问题，从而避免维度灾难。并且特征过多会导致过拟合。 
SelectPercentile：选择最强大的 X% 特征（X 是参数）
SelectKBest：选择 K 个最强大的特征（K 是参数）

###### 例子 
```
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

data = load_iris()
# print(data.data[0:5])

X = data.data
y = data.target

# select top 2 features
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)

# select top 10% features
X_new_percent = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
print(X_new_percent.shape)
```
#### 2.4 方法

![方法](https://ws3.sinaimg.cn/large/006tKfTcgy1fo6ne0nylcj318y0man0m.jpg)

#### 2.5 过滤filtering

- 缺点：缺乏反馈。未与学习算法结合，不能考虑到学习偏差。 - 
- 优点：水到渠成，步步分工，速度快

1. 对于过滤，决策树是一种比较好的过滤算法，运用ID3类似的算法可以根据信息增益的大小来过滤掉信息增益较小的特征，从而达到筛选的效果。 但是我们不需要得到决策树的最后结果，只是使用了它的筛选功能。筛选出来的特征我们可以导入其他的学习算法进一步得到结果。

2. 除了决策树删除信息增益较小的特征，还可以通过消除冗余的特征，比如：特征X2的效果=X1+X3的效果，则X2就是冗余的，可以删除。

#### 2.6 封装wrapping
- 缺点：速度慢 
- 优点：关注学习问题本身，特征筛选准确度更高

##### 2.6.1 前向搜索
首先逐个评估，利用选择排序的思想进行最优排序，但是不同的是每一次循环过后对已添加的特征整体进行一次评估，如果此次整体并没有比上次整体评估更好，则停止，选择上一个整体特征。

##### 2.6.2 反向搜索
类似一个团队逐步筛选人，每次循环筛选最不重要的一个人，不停的循环直到删除任何一个人时都会对精确度造成很大影响，则停止删除

### 3.0 特征转换


### 4.0 pca代码实现
```
def pca(dataMat, topNfeat=9999999):
    """pca

    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
    """

    # 计算每一列的均值
    meanVals = mean(dataMat, axis=0)
    # dataMat: (1567L, 590L)
    print 'dataMat:', shape(dataMat)
    # print 'meanVals', meanVals

    # 每个向量同时都减去 均值
    meanRemoved = dataMat - meanVals
    # print 'meanRemoved=', meanRemoved

    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    '''
    方差：（一维）度量两个随机变量关系的统计量
    协方差： （二维）度量各个维度偏离其均值的程度
    协方差矩阵：（多维）度量各个维度偏离其均值的程度

    当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
    当 cov(X, Y)<0时，表明X与Y负相关；
    当 cov(X, Y)=0时，表明X与Y不相关。
    '''
    covMat = cov(meanRemoved, rowvar=0)
    print 'covMat:', shape(covMat)
    # eigVals为特征值， eigVects为特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print 'eigVals=', eigVals
    # print 'eigVects=', eigVects
    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # 特征值的逆序就可以得到topNfeat个最大的特征向量

    # 特征值排序索引，从小到大
    eigValInd = argsort(eigVals)
    # print 'eigValInd1=', eigValInd

    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    # 特征索引
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print 'eigValInd2=', eigValInd
    # 重组 eigVects 最大到最小
    print 'eigVects:', shape(eigVects)
    redEigVects = eigVects[:, eigValInd]
    # (590L, 20L)
    print 'redEigVects=', shape(redEigVects)
    # 将数据转换到新空间
    # (1567L, 590L) (590L, 20L) (1L, 590L)
    print "---", shape(meanRemoved), shape(redEigVects),shape(meanVals)
    lowDDataMat = meanRemoved * redEigVects
    print "lowDDataMat:", shape(lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    # print 'lowDDataMat=', lowDDataMat
    # reconMat= (1567L, 590L)
    print 'reconMat=', shape(reconMat)
    return lowDDataMat, reconMat
```

### 5.0 正则化
该算法可以平衡特征的数量、精确度、泛化能力。
```
from sklearn import linear_model
clf = linear_model.Lasso()
clf.fit(features,labels)
print(clf.coef_)            #输出每个特征所占的比例，为0时说明被舍弃了
```
### 其他：pca效果
![pca_2](https://i.imgur.com/L9GeQ4b.png)

更高的 F1 得分，这意味着分类器的性能更高；