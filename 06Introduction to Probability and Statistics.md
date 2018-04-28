## Introduction to Probability and Statistics

### 01
#### 01.01 方差
在统计描述中，方差用来计算每一个变量（观察值）与总体均数之间的差异。为避免出现离均差总和为零，离均差平方和受样本含量的影响，统计学采用平均离均差平方和来描述变量的变异程度。

方差是**实际值与期望值**之差平方的平均值，而标准差是方差算术平方根。方差越大，数据的波动越大；方差越小，数据的波动就越小

![方差](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D113/sign=c388d5738013632711edc632a28ea056/023b5bb5c9ea15cee484a9a6bc003af33a87b233.jpg)

西格玛的平方 为总体方差, X为变量, u为总体均值, N为总体例数。
#### 01.02 方差的性质
![方差的性质](https://i.imgur.com/UXU6Swd.png)

#### 01.03 统计学意义

- 当数据分布比较分散（即数据在平均数附近波动较大）时，各个数据与平均数的差的平方和较大，方差就较大；当数据分布比较集中时，各个数据与平均数的差的平方和较小。因此**方差越大，数据的波动越大**；方差越小，数据的波动就越小。

- 标准差与方差不同的是，标准差和变量的计算单位相同，比方差清楚，因此很多时候我们分析的时候更多的使用的是标准差。

#### 01.04.01 协方差
X Y相互独立时，存在某种关系，协方差就是用来表示这种关系，用于衡量两个变量间的整体误差。
![协方差公式](http://img.my.csdn.net/uploads/201211/21/1353513364_9506.png)
(其中，E为数学期望或均值，D为方差，D开根号为标准差，E{ [X-E(X)] [Y-E(Y)]}称为随机变量X与Y的协方差，记为Cov(X,Y)，即Cov(X,Y) = E{ [X-E(X)] [Y-E(Y)]}，而两个变量之间的协方差和标准差的商则称为随机变量X与Y的相关系数，记为ρxy)

#### 结论
协方差为正，说明X,Y同向变化，协方差越大说明同向程度越高；如果协方差为负，说明X，Y反向运动，协方差越小说明反向程度越高。

#### 01.04.02 相关系数
相关系数也可以看成协方差：一种剔除了两个变量量纲影响、标准化后(归一化)的特殊协方差。

#### 结论
它消除了两个变量变化幅度的影响，而只是单纯反应**两个变量**每单位变化时的**相似程度**。

[如何通俗易懂地解释「协方差」与「相关系数」的概念？](https://www.zhihu.com/question/20852004)

#### 01.04.03 协方差矩阵
协方差矩阵计算的是不同维度之间的协方差。

![协方差矩阵1](https://i.imgur.com/SLDh3Dh.png)

参考：[对于概率论数字特征的理解](xueshu.baidu.com "对于概率论数字特征的理解")

#### 01.05 正态分布
![正太分布](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268/sign=91b83dc8b2119313c743f8b65d390c10/4ec2d5628535e5dd25ea56d576c6a7efcf1b62f2.jpg)

##### 01.05.01 图形特征
![正态分布图](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike150%2C5%2C5%2C150%2C50/sign=884edb64b63eb13550cabfe9c777c3b6/a5c27d1ed21b0ef413916fd2d7c451da80cb3ec4.jpg)

- 集中性：正态曲线的高峰位于正中央，即均数所在的位置。
- 对称性：正态曲线以均数为中心，左右对称，曲线两端永远不与横轴相交。
- 均匀变动性：正态曲线由均数所在处开始，分别向左右两侧逐渐均匀下降。

##### 01.05.02 固定尺度参数σ,位置参数μ变化时
固定尺度参数σ,位置参数μ变化时，f(x)图形的形状不变，只是沿着x轴作平移变换
![固定尺度参数σ,位置参数μ变化时](http://img.my.csdn.net/uploads/201212/18/1355793255_6746.jpg)

##### 01.05.03 位置参数μ固定,尺度参数σ变化时
位置参数μ固定,尺度参数σ变化时, f(x)图形的对称轴不变，形状在改变，越小，图形越高越瘦，越大，图形越矮越胖

![位置参数μ固定,尺度参数σ变化时](http://img.my.csdn.net/uploads/201212/18/1355793440_3003.jpg)

##### 01.05.04 正偏 负偏
![正偏态](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike60%2C5%2C5%2C60%2C20/sign=6d2f792bb2014a9095334eefc81e5277/5882b2b7d0a20cf4429538467d094b36acaf9926.jpg)

![负偏态](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike60%2C5%2C5%2C60%2C20/sign=59743db29616fdfacc61cebcd5e6e731/78310a55b319ebc449deb0008926cffc1f171681.jpg)

#### 01.06 隐马尔可夫模型
隐马尔可夫模型是统计模型，它用来描述一个含有隐含未知参数的马尔可夫过程。其难点是从可观察的参数中确定该过程的隐含参数，然后利用这些参数来作进一步的分析，例如模式识别。

#### 01.07 二项分布
**离散型**随机变量的概率分布
二项分布是指统计变量中只有性质不同的两项群体的概率分布。所谓两项群体是按两种不同性质划分的统计变量，是二项试验的结果。即各个变量都可归为两个不同性质中的一个，两个观测值是对立的。因而两项分布又可说是两个对立事件的概率分布。

期望：Eξ=np；
方差：Dξ=npq；
其中q=1-p

#### 01.08 各种分布的比较

#### 01.09 全概率公式
![全概率公式](https://i.imgur.com/lFpdB5j.png)

#### 例子
射击选手选拔

#### 01.10 t分布
![t分布](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=e240ada2d2a20cf4529df68d17602053/80cb39dbb6fd52666bd41631a918972bd4073613.jpg)

##### 01.10.01 特征
- 以0为中心，左右对称的单峰分布；
- t分布是一簇曲线，其形态变化与n（确切地说与自由度df）大小有关。自由度df越小，t分布曲线越低平；自由度df越大，t分布曲线越接近标准正态分布（u分布）曲线；
- 随着自由度逐渐增大，t分布逐渐接近标准正态分布；

##### 01.10.02 正态分布曲线与t曲线有何不同?
t分布与正态分布一样，是一个单峰对称呈钟形的分布，其对称轴通过分布的平均，数t分布曲线在正负两个方向上也以横轴为它的渐近线。

与正态分布相比，t分布曲线中间低而尖峭，两头高而平缓。t分布的最大特点是它实质上是一族分布，每一个t分布的形态受一个称为自由度的指标所制约。对应一个自由度就有一个t分布，随着自由度的增大，t分布曲线的中间就越来越高，两头却越来越低，整条曲线越来越趋近于正态分布，当自由度接近无穷大时，t分布就变成了正态分布。

#### 01.11 gamma分布
![gamma分布](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=b4fcd4988db1cb132a643441bc3d3d2b/a8773912b31bb051784d0c5f367adab44aede058.jpg)

指数分布是伽马分布的一种特殊形式;

### 02.01 中心极限定理
#### 02.01.01 独立同分布的中心极限定理
设随机变量X1，X2，......Xn，......独立同分布，并且具有有限的数学期望和方差：E(Xi)=μ，D(Xi)=σ20(k=1,2....)，则对任意x，分布函数

![中心极限定理](https://gss3.bdstatic.com/7Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D448/sign=559d641a11d5ad6eaef965eeb9ca39a3/aec379310a55b3196c42ddab49a98226cefc17b0.jpg)

该定理说明，当n很大时，随机变量![变量](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D126/sign=81f78d20114c510faac4e61856582528/77094b36acaf2edda57f1a01861001e93801934f.jpg) 近似地服从标准正态分布N(0，1)。因此，当n很大时，  ![变形](https://gss0.bdstatic.com/-4o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D146/sign=be5e9e3d5d2c11dfdad1bb2755266255/d62a6059252dd42af4b35875093b5bb5c8eab8b2.jpg)近似地服从正态分布N(nμ，nσ2)．


#### 02.01.02 棣莫佛－拉普拉斯定理
设随机变量X(n=1,2,...,)服从参数为n，p(0<p<1)的二项分布，则对于任意有限区间(a，b)有

![](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D324/sign=9859d880d7c451daf2f60ae982fc52a5/8326cffc1e178a823a8d3f7efc03738da877e8bf.jpg)

该定理表明，正态分布是二项分布的极限分布，当数充分大时，我们可以利用上式来计算二项分布的概率。

[中心极限定理](https://baike.baidu.com/item/%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86/829451?fr=aladdin "中心极限定理")
[wiki中心极限定理](http://wiki.mbalib.com/wiki/%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86 "wiki中心极限定理")

### 02.02 对数
![对数](https://gss0.bdstatic.com/-4o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike116%2C5%2C5%2C116%2C38/sign=9de347ead52a6059461de948495d5ffe/0dd7912397dda14461d8c791b0b7d0a20df486c4.jpg)

### 02.02.01 对数的运算性质
![对数的运算性质](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike116%2C5%2C5%2C116%2C38/sign=91e60b010b4f78f0940692a118586130/4bed2e738bd4b31ced0a01ee8fd6277f9e2ff8b5.jpg)

### 02.03 三角函数
![三角函数公式1](https://i.imgur.com/QUxzlXb.png)

![三角函数公式2](https://i.imgur.com/yEZOFtC.png)

![三角函数公式3](https://i.imgur.com/jNcYGik.png)

#### 02.03.01 奇变偶不变，符号看象限
![奇变偶不变，符号看象限](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike72%2C5%2C5%2C72%2C24/sign=b24f03ab5566d0166a14967af642bf62/f31fbe096b63f624e0f4b9658044ebf81a4ca32f.jpg))

#### 02.03.02 二角和差公式
![二角和差公式1](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D192/sign=e05915b649166d223c77119d74220945/faf2b2119313b07e7f22c27506d7912397dd8c34.jpg)

![二角和差公式2](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D192/sign=e05915b649166d223c77119d74220945/faf2b2119313b07e7f22c27506d7912397dd8c34.jpg)

#### 02.03.03 证明

![证明1](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D220/sign=d6a859052b34349b70066987f9eb1521/55e736d12f2eb93868d5c2edd6628535e5dd6f71.jpg)

![证明2](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D220/sign=32b52ca484d6277fed12353a18391f63/7acb0a46f21fbe09ded77b0b68600c338744ad20.jpg)

### 03 似然估计
#### 03.01 极大似然估计
极大似然估计，通俗理解来说，就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值！

换句话说，极大似然估计提供了一种给定观察数据来评估模型参数的方法，即："模型已定，参数未知"。

`p(x|θ)`输入有两个：`x`表示某一个具体的数据； `θ`表示模型的参数:
1. 如果`θ`是已知确定的，`x`是变量，这个函数叫做概率函数(`probability function`)，它描述对于不同的样本点x，其出现概率是多少。

2. 如果`x`是已知确定的，`θ`是变量，这个函数叫做似然函数(`likelihood function`), 它描述对于不同的模型参数，出现`x`这个样本点的概率是多少。

极大似然估计中采样需满足一个重要的假设，就是**所有的采样都是独立同分布**的。

参考：[一文搞懂极大似然估计](https://zhuanlan.zhihu.com/p/26614750 "一文搞懂极大似然估计")

#### 03.02 对数化的似然函数
似然函数的表达式通常出现连乘：

![log_likelihood_1.png](https://i.imgur.com/ZE0gH1x.png)

对多项乘积的求导往往比较复杂，但是对于多项求和的求导却要简单的多，**对数函数不改变原函数的单调性和极值位置**，而且根据对数函数的性质可以将乘积转换为加减式，这可以大大简化求导的过程：

![log_likelihood_2.png](https://i.imgur.com/8xKb1UK.png)

参考：[似然与极大似然估计](https://zhuanlan.zhihu.com/p/22092462 "似然与极大似然估计")

