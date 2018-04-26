## K-means
### 1.0 K-means
它可以发现 K 个不同的簇, 且每个簇的中心采用簇中所含值的均值计算而成.簇个数 K 是用户指定的, 每一个簇通过其质心（centroid）, 即簇中所有点的中心来描述.

##### 优点
容易实现
##### 缺点
1. 收敛太慢
2. 算法复杂度高O(nkt)
3. 不能发现非凸形状的簇，或大小差别很大的簇
4. 需样本存在均值（限定数据种类）
6. 需先确定k的个数
7. 对噪声和离群点敏感
8. 最重要是结果不一定是全局最优，只能保证局部最优。


使用数据类型 : 数值型数据

#### 1.1 Random Initialization
![K-means_random_initialization_1](https://i.imgur.com/mNSNn6i.png)

1. 簇中心小于训练集个数
2. 随机初始化簇中心

#### 1.2  K-means的局限
1. Kmeans是爬山算法,它非常**依赖于你的初始聚类中心所处的位置**，所以同一个训练集训练出的模型，可能预测出不一样的结果；

2. 局部最小值(理由同上)

![kmeans局部最小](https://i.imgur.com/wvXiaBV.png)

3. 差的局部最小值

![kmeans差的局部最小值](https://i.imgur.com/CIZnV0A.png)

#### 1.3 避免局部最优的方法
![K-means_avoid_local_optimization_1](https://i.imgur.com/TIozItH.png)

循环100次K-means方法，取最小的代价函数对应的分类


### 2.0 K-Means伪代码实现
```
1. 创建 k 个点作为起始质心（通常是随机选择）
2. 当任意一个点的簇分配结果发生改变时
    2.1 对数据集中的每个数据点
        2.1.1 对每个质心
        2.1.2 计算质心与数据点之间的距离
        2.1.3 将数据点分配到距其最近的簇
    2.2 对每一个簇, 计算簇中所有点的均值并将均值作为质心
```
#### 2.1 Choosing the Number of Clusters
手动选择聚类的数目

##### 2.1.1 肘部法则(Elbow Method)
![K-means_elbow_method_1](https://i.imgur.com/DCjPL2Q.png)

左图，当`K=3`时，类簇指标的下降趋势最快，所以应该取`K=3`；

如果遇到5比3代价函数高，那么5可能陷入局部最优，应该重新初始化`K=5`的簇中心，然后计算代价函数；

##### 2.1.2 轮廓系数
![silhouette_coefficient_1.jpg](https://i.imgur.com/SxSKYCC.jpg)

![silhouette_coefficient_2.jpg](https://i.imgur.com/8aoW8EF.jpg)

1. `a(i)`表示 点`i`到当前簇其他点的平均距离。
2. `b(i)`表示 点`i`到每一个非当前簇点的所有样本平均距离，再求这些平均距离的最小值。

##### 判断
1. `si`接近1，则说明样本`i`聚类合理；
2. `si`接近-1，则说明样本``i更应该分类到另外的簇；
3. 若`si`近似为0，则说明样本`i`在两个簇的边界上。

##### 2.1.3 Calinski-Harabasz准则
`Calinski-Harabasz`分数值`s`的数学计算公式：

![Calinski-Harabasz_1.png](https://i.imgur.com/87WXlNj.png)

其中`m`为训练集样本数，`k`为类别数。 `Bk`为类别之间的协方差矩阵，`Wk`为类别内部数据的协方差矩阵。`tr`为矩阵的迹。

也就是说，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的`Calinski-Harabasz`分数会高。在`scikit-learn`中，`Calinski-Harabasz Index`对应的方法是`metrics.calinski_harabaz_score`


#### 2.2 K-means optimization object
![K-means_optimization_objective_1](https://i.imgur.com/pQTG1Qd.png)

- `uc(i)`:表示`x(i)`分给的那个`cluster`的`cluster centroid`
- `K`表示有`K`个`cluster`, `k`表示`cluster centoid`的`index`
- `cost function`为`x(i)`到属于它的`cluster`的`cluster centroid`的距离的平方的累加

通过求`cost function`的最小值来求得参数`c`与`u`，这个`cost function`有时也称为`distortion cost function`(失真代价函数)


参考：

1. [用scikit-learn学习K-Means聚类](http://www.cnblogs.com/pinard/p/6169370.html "用scikit-learn学习K-Means聚类")
2. [k-means的k值该如何确定？](https://www.zhihu.com/question/29208148 "k-means的k值该如何确定？")

### 3.1 二分 K-Means 聚类算法
背景：因为K-means可能偶尔会陷入局部最小值

#### 3.1.1 二分K-Means伪代码实现(机器学习实战)
```
1. 将所有点看成一个簇
2. 当簇数目小雨 k 时
3. 对于每一个簇
    3.1 计算总误差
    3.2 在给定的簇上面进行 KMeans 聚类（k=2）
    3.3 计算将该簇一分为二之后的总误差
4. 选择使得误差最小的那个簇进行划分操作
```

### 3.2 ISODATA算法
### 3.2.1 ISODATA算法框架
![ISODATA算法框架](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025949447-680611657.png)

#### 3.2.2 第5步中的分裂操作和第6步中的合并操作
![ISODATA合并](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025951775-1194408309.png)

#### 3.2.3 ISODATA分裂
![ISODATA分裂](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025954494-895315300.png)

### 3.3 K-Means++
这个算法也只是对初始点的选择有改进而已，其他步骤都一样。初始质心选取的基本思路就是，**初始的聚类中心之间的相互距离要尽可能的远**。

#### 3.3.0 K-Means++ 实现步骤
![K-Means++](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025934541-260409014.png)

#### 3.3.1 K-Means++实例
![样本分布](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025944728-1116870094.png)

假设经过图2的步骤一后6号点被选择为第一个初始聚类中心，那在进行步骤二时每个样本的D(x)和被选择为第二个聚类中心的概率如下表所示：

![K-Means++概率计算](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025946338-787569010.png)

其中的P(x)就是每个样本被选为下一个聚类中心的概率。最后一行的Sum是概率P(x)的累加和，用于轮盘法选择出第二个聚类中心。方法是随机产生出一个0~1之间的随机数，判断它属于哪个区间，那么该区间对应的序号就是被选择出来的第二个聚类中心了。例如1号点的区间为[0,0.2)，2号点的区间为[0.2, 0.525)。

[K-means聚类算法的三种改进(K-means++,ISODATA,Kernel K-means)介绍与对比](https://www.cnblogs.com/yixuan-xu/p/6272208.html "K-means聚类算法的三种改进(K-means++,ISODATA,Kernel K-means)介绍与对比")


### 4.0 单连锁聚类
