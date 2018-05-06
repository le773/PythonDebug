### 2.0 特征选择
#### 2.1 相关性
相关性是指该特征与`label`的判断是否有帮助，有帮助则是相关的，无帮助就是不相关的。而相关性的强弱可以通过该特征的必要程度来决定。 

##### 例子 特征ABCD 
`A+B`可以判断`label`,`A+C`也可以判断`label`
可以说`A`的相关性强，因为它不可替代；而`B`和`C`的相关性弱，因为它们之间可以互相替代。`D`则是不相关。

#### 2.2 有用性
是指一个特征虽然不相关，但是它在某一个特定算法中会起到比较大的作用，则指这个特征是有用的，具有有用性。


#### 2.3 特征增加

1. 根据自己经验直觉觉得哪个`feature`可以用来训练
2. `code the feature`
3. 可视化`feature`
4. 重复该步骤

#### 2.3 特征删除

尽量减少特征，简化学习问题，从而避免维度灾难。并且特征过多会导致过拟合。 
`SelectPercentile`：选择最强大的 `X%` 特征（`X` 是参数）
`SelectKBest`：选择 `K` 个最强大的特征（`K` 是参数）

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

# TODO：反向转换中心点
log_centers = pca.inverse_transform(X_new)
```
#### 2.4 方法

![方法](https://ws3.sinaimg.cn/large/006tKfTcgy1fo6ne0nylcj318y0man0m.jpg)

#### 2.5 过滤filtering
过滤式方法先对数据集进行特征选择，然后再训练学习器，特征选择过程与后续学习器无关。这相当于先用特征选择过程对初始特征进行“过滤”，再用过滤后的特征来训练模型。

特征子集的重要性由子集中每个特征所对应的相关统计量分量之和来决定。

Relief是过滤式特征选择方法：

给定训练集T={(x<sub>1</sub>,y<sub>1</sub>),(x<sub>2</sub>,y<sub>2</sub>),...,(x<sub>m</sub>,y<sub>m</sub>)}，对每一示例x<sub>i</sub>。Relief先在x<sub>i</sub>的同类样本中寻找其最近邻x<sub>i,nh</sub>，称为“猜中近邻”(near-hit)，

再从x<sub>i</sub>的异类样本中寻找其最近邻x<sub>i,nm</sub>，称为“猜错近邻”(near-miss)，

然后，相关统计量对应于属性j的分量为：

![feature_filter_1.png](https://i.imgur.com/edn3Vwi.png)

其中x<sup>j</sup><sub>a</sub>表示样本x<sub>a</sub>在属性j上的取值，diff(x<sup>j</sup><sub>a</sub>,x<sup>j</sup><sub>b</sub>)取决于属性j的类型：

若属性j为离散型，则x<sup>j</sup><sub>a</sub>=x<sup>j</sup><sub>b</sub>时，diff(x<sup>j</sup><sub>a</sub>,x<sup>j</sup><sub>b</sub>)=0，否则为1；

若属性j为连续型，则diff(x<sup>j</sup><sub>a</sub>,x<sup>j</sup><sub>b</sub>)=|x<sup>j</sup><sub>a</sub> - x<sup>j</sup><sub>b</sub>|，

注意x<sup>j</sup><sub>a</sub>，x<sup>j</sup><sub>b</sub>已规范化到[0,1]区间。

从上述公式可以看出，若x<sub>i</sub>与其猜中近邻x<sub>i,nh</sub>在属性j上的距离小于x<sub>i</sub>与其猜错近邻x<sub>i,nm</sub>，则说明j对区分同类与异类样本是有益的，于是增大属性j所对应的统计量分量；反之，则说明属性j起负面作用，于是减小属性j所对应的统计量分量。最后对基于不用样本得到的估计结果进行平均，就得到各属性的相关统计分量，分量值越大，则说明对应属性分类能力越强。

Relief-F是Relief的变体，能处理多分类问题。

假定数据集D中的样本来自|y|个类别。对示例x<sub>i</sub>，若它属于第k类(kε{1,2,..,|y|})，则Relief-F先在第k类的样本中寻找x<sub>i</sub>的最近示例x<sub>i,nh</sub>并将其作为猜中近邻。

然后在第k类之外的每个类中找到一个x<sub>i</sub>的最近邻作为猜错近邻，记为x<sub>i,nm</sub>(l=1,2,..,|y|; l neq k)。于是，相关统计量对应于属性j的分量为

![feature_filter_2.png](https://i.imgur.com/8URwswy.png)

其中p<sub>l</sub>为第l类样本在数据集D中所占的比例。

- 缺点：缺乏反馈。未与学习算法结合，不能考虑到学习偏差。 - 
- 优点：水到渠成，步步分工，速度快

1. 对于过滤，决策树是一种比较好的过滤算法，运用ID3类似的算法可以根据信息增益的大小来过滤掉信息增益较小的特征，从而达到筛选的效果。 但是我们不需要得到决策树的最后结果，只是使用了它的筛选功能。筛选出来的特征我们可以导入其他的学习算法进一步得到结果。

2. 除了决策树删除信息增益较小的特征，还可以通过消除冗余的特征，比如：特征`X2的效果=X1+X3的效果`，则`X2`就是冗余的，可以删除。

#### 2.6 封装wrapping

与过滤式特性选择不考虑后续学习器不同，包裹式特征选择直接把最终要使用的学习器的性能作为特征子集的评价准则。换言之，包裹式特征选择的目的就是给学习器选择最有利于其性能，“量身定做”的特征子集。

从最终学习器性能来看，包裹式特征选择比过滤式特征选择更好，

另一方面，由于在特征选择过程中需要多次训练学习器，因此包裹式特征选择的计算开销通常比过滤式选择要大得多。

LVM是一个典型的包裹式特征选择方法

![feature_wrapper_1.png](https://i.imgur.com/3pL8CF8.png)

##### 2.6.1 前向搜索
首先逐个评估，利用选择排序的思想进行最优排序，但是不同的是每一次循环过后对已添加的特征整体进行一次评估，如果此次整体并没有比上次整体评估更好，则停止，选择上一个整体特征。

##### 2.6.2 反向搜索
类似一个团队逐步筛选人，每次循环筛选最不重要的一个人，不停的循环直到删除任何一个人时都会对精确度造成很大影响，则停止删除


#### 2.7 嵌入式选择
嵌入式特征选择是将特征选择过程与学习器训练过程融为一体，两者在同一个优化过程中完成，即学习器训练过程中自动进行了特征选择。




### 3.0 特征转换
