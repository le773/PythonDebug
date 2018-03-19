### 1.0 PCA
Principal Components Analysis:主成分分析
主成分是由数据中具有最大方差的方向决定的，因为可以最大程度的保留信息量。

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
1. 可视化高维数据
1. 怀疑数据中存在噪声，可以帮助抛弃重要性小的特征，去除噪声
1. 让算法在少量的特征上更有效，比如说人脸识别，可以将维度降低1/10，然后使用svm来进行训练，之后发现被拍摄人的身份

### 2.0 pac代码实现
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


def replaceNanWithMean():
    datMat = loadDataSet(os.getcwd() + "\\codetwo\\" + 'input/13.PCA/secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[nonzero(isnan(datMat[:, i].A))[0],i] = meanVal
    return datMat
```

### 其他：pca效果
![pca_2](https://i.imgur.com/L9GeQ4b.png)