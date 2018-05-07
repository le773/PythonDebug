### 1.0 PCA
`Principal Components Analysis`:主成分分析

主成分是由数据中具有最大方差的方向决定的，因为可以最大程度的保留信息量。

`PCA`擅长用于旋转、偏移坐标轴，使之达到降维的作用；
`PCA`将几个相关的特征用一个特征来表示，即将多维降为一维；

其实我们不必知道那几个特征相关，我们可以直接把许多特征丢进`PCA`，`PCA`会自动帮我们分析哪些特征可以合并降维。 

![pca旋转降维](http://img.blog.csdn.net/20160207114645575)

#### 1.1 是否可以舍弃特征？
`pca`会找出一种方法，来将各种输入特征中的信息合并起来；所以，如果在进行`PCA`之前排除所有输入特征，某种程度上讲，也就排除`PCA`能够挽救的信息,(如果有很多不相关的特征，是可以舍弃的，但是要小心)；制作主成分之后对其进行特征选择是可以的。

#### 1.2.1 PCA算法理论推导
![pca_1.png](https://i.imgur.com/nMjUa9l.png)

矩阵X通过某个线性变换得出的新矩阵B的每一列都正交，用矩阵表示即为：

B<sup>T</sup> x B = D

其中，D是一个对角阵。 那么我们假设这个变换是XM=B，将这个带入得：

(XM)<sup>T</sup> x (XM) = D

M<sup>T</sup> x X<sup>T</sup> x X x M = D

X<sup>T</sup> x X = (M<sup>T</sup>)<sup>-1</sup> x D x (X<sup>T</sup>)<sup>-1</sup>

而X<sup>T</sup>X是一个对角阵，那么它的特征值分解X<sup>T</sup>X=VDV<sup>-1</sup>中的V是正交单位阵，那么有V<sup>T</sup>=V<sup>-1</sup>，那么这个V就满足对M的要求。

#### 1.2.2 PCA算法伪代码
![svd_15.png](https://i.imgur.com/W7c21LY.png)

![pca_1.png](https://i.imgur.com/nqdbRUz.png)

#### 1.3 PCA的特点
- `PCA`将输入特征转化为主成分的系统化方式
- 这些主成分可以作为新特征使用
- 主成分的定义是数据中会使方差最大化的方向，将出现信息丢失的可能性降至最低
- 可以对主成分划分等级，数据因特定主成分而产生的方差越大，那么该主成分的级别越高，方差最大的主成分即为第一个主成分
- 主成分在某种意义上是**相互垂直**的，第二个主成分基本不会与与第一个主成分重叠，可以将他们作为单独的特征对待
- 主成分数量有限，最大值等于数据集中的输入特征数量

结论1：协方差矩阵（或X<sup>T</sup>X）的奇异值分解结果和特征值分解结果一致。

#### 1.4 什么时候使用PCA？
- 想要访问隐藏的特征，这些特征可能隐藏显示在你的数据图案中
- 降维

a. 可视化高维数据

b. 怀疑数据中存在噪声，可以帮助抛弃重要性小的特征，**去除噪声**

c. 让算法在少量的特征上更有效，比如说人脸识别，可以将维度降低`1/10`，然后使用`svm`来进行训练，之后发现被拍摄人的身份

d. 防止过拟合

- 提高算法效率、精确度

#### 1.5 选择主成分
##### PCA过拟合
当使用大量 `PCA` 时，会看到任何过拟合迹象, 性能开始下降。

##### 主要原则
主要原则：必须最大程度的保留原信息的数据。

##### 停止条件
通过增加主成分，观察`f1`得分(等)，如果出现停滞(下降)则停止增加。

![主成分](http://img.blog.csdn.net/20171206233759666?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhuaW5nMTJM/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

通过`PCA`还可以得出这条轴的一个散布值

如果散布率较小，该散布值对于第一条变量轴来说倾向于一个很大的值，而对于第二条变量轴来说则小得多 

所以这个数字，碰巧是一个特征值，结果来自于`PCA`执行的一种特征值分解；

#### 1.6 轴何时占主导地位

![轴何时占主导地位](http://img.blog.csdn.net/20171206235513061?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhuaW5nMTJM/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### 长轴是否占优势？
长轴的重要值，特征值远远大于短轴的特征值。

#### 1.7 最大方差与信息损失
![信息损失](https://i.imgur.com/AeYD61A.png)

丢失的信息就是数据中的点到主成分方向的距离，所以我们要寻找的就是总距离最小时主轴的方向，即主成分方向。

根据数学知识数据最大方差的方向就是总距离最小的方向(实际上将点与其在该线上的投影之间的距离最小化(信息丢失最小))，即主成分方向就是由该数据最大方差的方向。

![信息损失2](https://i.imgur.com/2kcb8ki.png)

上图中绿色点的损失大于黄色点。

#### 1.8 主成分

### 2.0 pca代码实现
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

### 其他：pca效果
![pca_2](https://i.imgur.com/L9GeQ4b.png)

更高的`F1`得分，这意味着分类器的性能更高；
