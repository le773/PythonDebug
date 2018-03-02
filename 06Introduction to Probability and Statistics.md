###### Introduction to Probability and Statistics

### 01
#### 01.01 方差
在统计描述中，方差用来计算每一个变量（观察值）与总体均数之间的差异。为避免出现离均差总和为零，离均差平方和受样本含量的影响，统计学采用平均离均差平方和来描述变量的变异程度。

方差是**实际值与期望值**之差平方的平均值，而标准差是方差算术平方根。

![方差](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D113/sign=c388d5738013632711edc632a28ea056/023b5bb5c9ea15cee484a9a6bc003af33a87b233.jpg)

西格玛的平方 为总体方差, X为变量, u为总体均值, N为总体例数。
#### 01.02 方差的性质
![方差的性质](https://i.imgur.com/UXU6Swd.png)

#### 01.03 统计学意义

- 当数据分布比较分散（即数据在平均数附近波动较大）时，各个数据与平均数的差的平方和较大，方差就较大；当数据分布比较集中时，各个数据与平均数的差的平方和较小。因此**方差越大，数据的波动越大**；方差越小，数据的波动就越小。

- 标准差与方差不同的是，标准差和变量的计算单位相同，比方差清楚，因此很多时候我们分析的时候更多的使用的是标准差。

#### 01.04 全概率公式
![全概率公式](https://i.imgur.com/lFpdB5j.png)

###### 例子
射击选手选拔

#### 01.05 正态分布
![正太分布](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268/sign=91b83dc8b2119313c743f8b65d390c10/4ec2d5628535e5dd25ea56d576c6a7efcf1b62f2.jpg)
##### 图形特征
![正态分布图](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike150%2C5%2C5%2C150%2C50/sign=884edb64b63eb13550cabfe9c777c3b6/a5c27d1ed21b0ef413916fd2d7c451da80cb3ec4.jpg)
- 集中性：正态曲线的高峰位于正中央，即均数所在的位置。
- 对称性：正态曲线以均数为中心，左右对称，曲线两端永远不与横轴相交。
- 均匀变动性：正态曲线由均数所在处开始，分别向左右两侧逐渐均匀下降。

#### 01.06 隐马尔可夫模型
隐马尔可夫模型是统计模型，它用来描述一个含有隐含未知参数的马尔可夫过程。其难点是从可观察的参数中确定该过程的隐含参数，然后利用这些参数来作进一步的分析，例如模式识别。