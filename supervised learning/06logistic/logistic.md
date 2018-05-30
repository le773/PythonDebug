### 1.0 导论
#### 1.1 概括
逻辑回归假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。

#### 1.2 信息熵

熵越大，信息量越大，越不确定(均匀分布)，计算熵的公式在等概时有最大值。

#### 1.3 sigmoid 优缺点
优点：

1. 在于输出范围有限，所以数据在传递的过程中不容易发散。
2. 输出范围为(0, 1)，所以可以用作输出层，输出表示概率。
3. 求导容易 `y=sigmoid(x), y'=y(1-y)`

缺点：

1. 饱和的时候梯度太小。

### 逻辑回归的优缺点
优点：

1.形式简单，模型的可解释性非常好。

2.资源占用小，尤其是内存。因为只需要存储各个维度的特征值。

3.训练速度较快。分类的时候，计算量仅仅只和特征的数目相关。并且逻辑回归的分布式优化sgd发展比较成熟，训练的速度可以通过堆机器进一步提高。

缺点：

1.很难处理数据不平衡的问题。

2.逻辑回归本身无法筛选特征。

3.处理非线性数据较麻烦。

4.准确率并不是很高。因为形式非常的简单(非常类似线性模型)，很难去拟合数据的真实分布。

----------

### 2.0 Logistic代价函数推导过程
`sigmoid`函数很适合用作二分类：

![logistic_cost_2.png](https://i.imgur.com/Z5WY55M.png)

#### 2.1 Logistic代价函数推导过程
假设样本为正的概率是：

![logistic_cost_3.png](https://i.imgur.com/yIgXIA0.png)

对于二元分类，符合伯努利分布（`the Bernoulli distribution`, 又称两点分布，0-1分布），因为二元分类的输出一定是0或1，典型的伯努利实验。所以假设样本预测为正负的概率：

![logistic_cost_1.png](https://i.imgur.com/t4T020p.png)

h<sub>θ</sub>(x):表示结果取1的概率；

1-h<sub>θ</sub>(x):表示结果取0的概率；

由最大似然函数可知：

![logistic_cost_10.png](https://i.imgur.com/890X41E.png)

上述式子综合起来：

P(y|x;θ)=(h<sub>θ</sub>(x))<sup>y</sup> \* (1- h<sub>θ</sub>(x))<sup>1-y</sup>

注：上标表示第几个样本，下标表示这个样本的第几个特征

取似然函数为：

![logistic_cost_4.png](https://i.imgur.com/TgfFqRr.png)

两边同时取对数：

![logistic_cost_5.png](https://i.imgur.com/VHT1HuN.png)

最大似然函数就是使l(θ)取最大时的θ，其实这里使用梯度上升求解，求得的θ就是要求的最佳参数。

方法一，因为求似然函数的最大值，所以采用梯度上升的方法：

![logistic_cost_6.png](https://i.imgur.com/xhswgxA.png)

由此可以看出最终的更新规则为：   

![logistic_cost_7.png](https://i.imgur.com/5XWyZ7W.png)

方法二、对此`logistic`的代价函数变形：

![logistic_cost_8.png](https://i.imgur.com/BucNDvH.png)

梯度下降法求的最小值，θ更新过程

![logistic_cost_13.png](https://i.imgur.com/5w5ommF.png)

![logistic_cost_14.png](https://i.imgur.com/0KXBuwH.png)

θ更新过程可以写成：

![logistic_cost_15.png](https://i.imgur.com/mo2MU4F.png)

#### 2.2 补充：代价函数推导过程
![logistic_cost_1.png](https://i.imgur.com/jKApTaG.png)

#### 2.3 Logistic回归正则化

![linear_cost_regularization_3.png](https://i.imgur.com/ujSroz1.png)

### 3.0 logistic 随机梯度上升
#### 3.1 随机梯度上升算法
```
# 随机梯度上升算法（随机化）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = range(m)
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4/(1.0+j+i)+0.0001    # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0,len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = classLabels[dataIndex[randIndex]] - h
            # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
    return weights
```

第一处改进为`alpha`的值。`alpha`在每次迭代的时候都会调整，这回缓解上面波动图的数据波动或者高频波动。另外，虽然`alpha`会随着迭代次数不断减少，但永远不会减小到 0，因为我们在计算公式中添加了一个常数项。

第二处修改为`randIndex`更新，这里通过随机选取样本拉来更新回归系数。这种方法将减少周期性的波动。这种方法每次随机从列表中选出一个值，然后从列表中删掉该值（再进行下一次迭代）。

#### 3.2 权重更新
```
weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
```
![Ridge.png](https://i.imgur.com/Lob12i4.png)

#### 3.4 Logistic 优缺点

优点: 计算代价不高，易于理解和实现。

缺点: 容易欠拟合，分类精度可能不高。

适用数据类型: 数值型和标称型数据。

### 4.0 补充问题
#### 4.1 `sigmoid`激活函数与双曲正切激活函数的关系
```
tanh(z) = 2σ(2z) − 1
```

#### 4.2 为什么 LR 模型要使用 sigmoid 函数，背后的数学原理是什么？
#### 4.2.1 指数族分布(the exponential family distribution)
若某概率分布满足:

P(y;η)=b(y) \* exp(η<sup>T</sup> \* T(y) - α(η))

就是指数分布。其中：

η是自然参数，T(y)是充分统计量，exp(-α(η))起到归一化作用。

统计学中，伯努利分布、高斯分布、多项式分布、泊松分布都是指数族分布。

而<font color=red>指数家族所具有的最佳性质，即最大熵的性质</font>，这是LR 模型要使用 sigmoid 函数原因之一。

#### 4.2.2 为什么 LR 模型要使用 sigmoid 函数？
LR是二分类问题，自然选择p(y|x;θ)服从伯努利分布。通过(2.1 Logistic代价函数推导过程)可知，概率模型可以写为：

P(y|x;θ)=(h<sub>θ</sub>(x))<sup>y</sup> \* (1- h<sub>θ</sub>(x))<sup>1-y</sup>

令:φ=h<sub>θ</sub>(x)，那么上述式子有如下推导过程：

P(y;η)</br>
= φ<sup>y</sup> \* (1-φ)<sup>(1-y)</sup></br>
= exp(ylogφ + (1-y)log(1-φ))</br>
= exp(ylogφ - ylog(1-φ) + log(1-φ))</br>
= exp(ylog(φ/(1-φ)) + log(1-φ))

把伯努利分布可以写成指数族分布的形式：

(1) T(y) = y

(2) η = log(φ/(1-φ))

(3) α(η) = -log(1-φ) = log(1+e<sup>η</sup>)

(4) b(y) = 1

解(2)得，φ = 1/(1+e<sup>-η</sup>)，即`logistic sigmoid`形式。

所以`sigmoid`，或者说指数族分布(包含伯努利)所具有的最大熵性质。

回过来看`logistic regression`，这里假设了什么呢？

首先，我们在建模预测 `Y|X`，并认为 `Y|X` 服从伯努利分布，所以我们只需要知道 `P(Y|X)`；

其次我们需要一个线性模型，所以 `P(Y|X) = f(wx)`。通过<font color=red>最大熵</font>原则推出的这个`f`，就是`sigmoid`。
    
参考：

[为什么LR模型要使用sigmoid函数，背后的数学原理是什么？](https://www.zhihu.com/question/35322351/answer/67193153)

[逻辑回归为什么使用Sigmod作为激活函数？](https://blog.csdn.net/cloud_xiaobai/article/details/72152480)

#### 4.3 Logistic回归做多分类和Softmax回归
1. 如果类别之间互斥，就用softmax

2. 如果类别之间有联系，使用K个LR更合适。

##### 4.3.1 第一种：根据每个类别，都建立一个二分类器

直接根据每个类别，都建立一个二分类器，带有这个类别的样本标记为1，带有其他类别的样本标记为0。假如我们有个类别，最后我们就得到了个针对不同标记的普通的`logistic`分类器。

对于二分类问题，我们只需要一个分类器即可，但是对于多分类问题，我们需要多个分类器才行。假如给定数据集![logistic_multi_classfy_1.png](https://i.imgur.com/EVWsC5H.png)，它们的标记![logistic_multi_classfy_2.png](https://i.imgur.com/oEV2bOL.png)，即这些样本有k个不同的类别。

1. 我们挑选出标记为![logistic_multi_classfy_3.png](https://i.imgur.com/adiolsT.png)的样本，将挑选出来的带有标记c的样本的标记置为1，将剩下的不带有标记的样本的标记置为0。

2. 然后就用这些数据训练出一个分类器，我们得到![logistic_multi_classfy_4.png](https://i.imgur.com/I2cJNnq.png)（表示针对标记的`logistic`分类函数）。

![one_vs_all_classfier_1.png](https://i.imgur.com/j3wwn15.png)

3. 按照上面的步骤，我们可以得到k个不同的分类器。

4. 针对一个测试样本，我们需要找到这个分类函数输出值最大的那一个，即为测试样本的标记：

![gif_01.gif](https://i.imgur.com/d8Bmn9F.gif)

##### 4.3.2 第二种：修改logistic回归的损失函数
修改`logistic`回归的损失函数，让其适应多分类问题。这个损失函数不再笼统地只考虑二分类非1就0的损失，而是具体考虑每个样本标记的损失。这种方法叫做`softmax`回归，即`logistic`回归的多分类版本。

#### 4.4 线性回归与逻辑回归
![logistic_linear_1.jpg](https://i.imgur.com/NxvVNZY.jpg)

`LogisticRegression`就是一个被`logistic`方程归一化后的线性回归。

联系：
1. 都是广义的线性回归；
2. 逻辑回归的模型本质上是一个线性回归模型，逻辑回归都是以线性回归为理论支持的；

区别：
1. 经典线性模型的优化目标函数是最小二乘，而逻辑回归则是似然函数；
2. 线性回归在整个实数域范围内进行预测，敏感度一致，而分类范围，需要在[0,1]。逻辑回归就是一种减小预测范围，将预测值限定为[0,1]间的一种回归模型，因而对于这类问题来说，逻辑回归的鲁棒性比线性回归的要好。

#### 4.5 logistic回归属于线性模型还是非线性模型？
`logistic`回归本质上是线性回归，只是在特征到结果的映射中加入了一个`sigmoid`函数，即先把特征线性求和，然后使用非线性的函数将连续值映射到0与1之间。

#### 4.6 逻辑回归的损失函数为什么要使用极大似然函数作为损失函数？
损失函数一般有四种，平方损失函数，对数损失函数，HingeLoss0-1损失函数，绝对值损失函数。将极大似然函数取对数以后等同于对数损失函数。在逻辑回归这个模型下，对数损失函数的训练求解参数的速度是比较快的。至于原因大家可以求出这个式子的梯度更新

![logistic_gb_1.png](https://i.imgur.com/hPamDTM.png)

个式子的更新速度只和x<sup>i</sup><sub>j</sub>，y<sup>i</sup><sub>j</sub>相关。和sigmod函数本身的梯度是无关的。这样更新的速度是可以自始至终都比较的稳定。

#### 4.7 为什么不选平方损失函数的呢？
其一是因为如果你使用平方损失函数，你会发现梯度更新的速度和sigmod函数本身的梯度是很相关的。sigmod函数在它在定义域内的梯度都不大于0.25。这样训练会非常的慢。

#### 4.8 bernoulli distribution
`bernoulli distribution`即二项分布

![Bernoulli_Distribution_2.png](https://i.imgur.com/Bew6rjz.png)

期望：`Eξ=np`； 方差：`Dξ=npq`； 其中`q=1-p`

#### 4.9 连续特征的离散化：在什么情况下将连续的特征离散化之后可以获得更好的效果？
0. 离散特征的增加和减少都很容易，易于模型的快速迭代；
1. 稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展；
2. 离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄>30是1，否则0。如果特征没有离散化，一个异常数据“年龄300岁”会给模型造成很大的干扰；
3. 逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合；
4. 离散化后可以进行特征交叉，由M+N个变量变为M*N个变量，进一步引入非线性，提升表达能力；
5. 特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问；
6. 特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。

### 5.0 logistic过拟合
![logistic_1.png](https://i.imgur.com/m7FfRDh.png)

如果特征过多，在训练集上学到的模型参数可能很好，但是泛化不行。

#### 5.1 logistic过拟合解决方法
![logistic_2.png](https://i.imgur.com/u0U2bQp.png)
```
1. 减少特征数量
    a. 人工选择特征
    b. 模型选择算法(pca、..)
2. 正则化
    a. 保留所有特征，但是减小参数θ
    b. 当有很多特性时，模型表现很好，每一个都对预测有一定的贡献。
```
#### 5.2 logistic 正则化 代价函数

![logistic_3.png](https://i.imgur.com/vN4YzyT.png)

λ是正则化系数

1.如果它的值很大，说明对模型的复杂度惩罚大，对拟合数据的损失惩罚小，这样不会过分拟合数据，在训练数据上的偏差较大，在未知数据上的方差较小，但是可能出现欠拟合。

![logistic_4.png](https://i.imgur.com/qUr7LWp.png)

2.如果它的值很小，说明比较注重对训练数据的拟合，在训练集上偏差会小，但是可能过拟合。

#### 5.3 logistic 梯度下降
![logistic_5.png](https://i.imgur.com/OR0pGVc.png)

![logistic_6.png](https://i.imgur.com/x0OdhVR.png)

正则化后的线性回归的Normal Equation公式

![logistic_7.png](https://i.imgur.com/oAhfHui.png)

其他优化算法

1. Conjugate gradient method(共轭梯度法)
1. Quasi-Newton method(拟牛顿法)
1. BFGS method
1. L-BFGS(Limited-memory BFGS)

参考：

1. [bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution "bernoulli distribution")

1. [Logistic Regression and SVM](http://charleshm.github.io/2016/03/LR-SVM/)

1. [逻辑回归（Logistic Regression）和SVM的联系以及Kernel](https://blog.csdn.net/jackie_zhu/article/details/52331047)

1. [逻辑回归和SVM的区别是什么？各适用于解决什么问题？](https://www.zhihu.com/question/24904422/answer/92164679)

1. [logistic回归属于线性模型还是非线性模型？](https://www.zhihu.com/question/30726036 "logistic回归属于线性模型还是非线性模型？")

1. [逻辑回归代价函数推导](https://blog.csdn.net/pakko/article/details/37878837)

1. [逻辑回归知识点总结](https://www.cnblogs.com/ModifyRong/p/7739955.html)

1. [连续特征的离散化：在什么情况下将连续的特征离散化之后可以获得更好的效果？](https://www.zhihu.com/question/31989952/answer/54184582)
