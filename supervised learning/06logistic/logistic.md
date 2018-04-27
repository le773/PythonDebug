逻辑回归多分类
### Logistic

优点: 计算代价不高，易于理解和实现。
缺点: 容易欠拟合，分类精度可能不高。
适用数据类型: 数值型和标称型数据。

### sigmoid
优点：

1. 在于输出范围有限，所以数据在传递的过程中不容易发散。
2. 输出范围为(0, 1)，所以可以用作输出层，输出表示概率。
3. 求导容易 `y=sigmoid(x), y'=y(1-y)`

缺点：

1. 饱和的时候梯度太小。

### 信息熵

熵越大，信息量越大，越不确定(均匀分布)，计算熵的公式在等概时有最大值。

----------

sigmoid函数

![logistic_cost_2.png](https://i.imgur.com/Z5WY55M.png)

![logistic_cost_3.png](https://i.imgur.com/yIgXIA0.png)

sigmoid激活函数与双曲正切激活函数的关系
```
tanh(z) = 2σ(2z) − 1
```

对于二元分类，符合伯努利分布（the Bernoulli distribution, 又称两点分布，0-1分布），因为二元分类的输出一定是0或1，典型的伯努利实验。所以：

![logistic_cost_1.png](https://i.imgur.com/t4T020p.png)

由最大似然函数可知：

![logistic_cost_4.png](https://i.imgur.com/TgfFqRr.png)

两边同时取对数：

![logistic_cost_5.png](https://i.imgur.com/VHT1HuN.png)

logistic的代价函数：

![logistic_cost_8.png](https://i.imgur.com/BucNDvH.png)

因为求似然函数的最大值，所以采用梯度上升的方法：

![logistic_cost_6.png](https://i.imgur.com/xhswgxA.png)

由此可以看出最终的更新规则为：   

![logistic_cost_7.png](https://i.imgur.com/5XWyZ7W.png)


### logistic 随机梯度上升
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

第一处改进为 alpha 的值。alpha 在每次迭代的时候都会调整，这回缓解上面波动图的数据波动或者高频波动。另外，虽然 alpha 会随着迭代次数不断减少，但永远不会减小到 0，因为我们在计算公式中添加了一个常数项。

第二处修改为 randIndex 更新，这里通过随机选取样本拉来更新回归系数。这种方法将减少周期性的波动。这种方法每次随机从列表中选出一个值，然后从列表中删掉该值（再进行下一次迭代）。

### 权重更新
```
weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
```
![Ridge.png](https://i.imgur.com/Lob12i4.png)

### 为什么 LR 模型要使用 sigmoid 函数，背后的数学原理是什么？
说到底源于sigmoid，或者说exponential family所具有的最佳性质，即maximum entropy的性质。

回过来看logistic regression，这里假设了什么呢？

首先，我们在建模预测 Y|X，并认为 Y|X 服从bernoulli distribution，所以我们只需要知道 P(Y|X)；

其次我们需要一个线性模型，所以 P(Y|X) = f(wx)。接下来我们就只需要知道 f 是什么就行了。而我们可以通过最大熵原则推出的这个f，就是sigmoid。

其实前面也有人剧透了bernoulli的exponential family形式，也即是 1/ (1 + e^-z)
    
[为什么LR模型要使用sigmoid函数，背后的数学原理是什么？](https://www.zhihu.com/question/35322351/answer/67193153)

### Logistic回归做多分类和Softmax回归
####第一种方式根据每个类别，都建立一个二分类器

直接根据每个类别，都建立一个二分类器，带有这个类别的样本标记为1，带有其他类别的样本标记为0。假如我们有个类别，最后我们就得到了个针对不同标记的普通的logistic分类器。

对于二分类问题，我们只需要一个分类器即可，但是对于多分类问题，我们需要多个分类器才行。假如给定数据集![logistic_multi_classfy_1.png](https://i.imgur.com/EVWsC5H.png)，它们的标记![logistic_multi_classfy_2.png](https://i.imgur.com/oEV2bOL.png)，即这些样本有k个不同的类别。

我们挑选出标记为![logistic_multi_classfy_3.png](https://i.imgur.com/adiolsT.png)的样本，将挑选出来的带有标记c的样本的标记置为1，将剩下的不带有标记的样本的标记置为0。然后就用这些数据训练出一个分类器，我们得到![logistic_multi_classfy_4.png](https://i.imgur.com/I2cJNnq.png)（表示针对标记的logistic分类函数）。

按照上面的步骤，我们可以得到个不同的分类器。针对一个测试样本，我们需要找到这个分类函数输出值最大的那一个，即为测试样本的标记：

![gif_01.gif](https://i.imgur.com/d8Bmn9F.gif)

#### 修改logistic回归的损失函数
第二种方式是修改logistic回归的损失函数，让其适应多分类问题。这个损失函数不再笼统地只考虑二分类非1就0的损失，而是具体考虑每个样本标记的损失。这种方法叫做softmax回归，即logistic回归的多分类版本。


----------

### 线性回归与逻辑回归
![logistic_linear_1.jpg](https://i.imgur.com/NxvVNZY.jpg)

LogisticRegression 就是一个被logistic方程归一化后的线性回归。

### logistic回归属于线性模型还是非线性模型？
logistic回归本质上是线性回归，只是在特征到结果的映射中加入了一个sigmoid函数，即先把特征线性求和，然后使用非线性的函数将连续值映射到0与1之间。

[logistic回归属于线性模型还是非线性模型？](https://www.zhihu.com/question/30726036 "logistic回归属于线性模型还是非线性模型？")

### bernoulli distribution
`bernoulli distribution`即二项分布

![Bernoulli_Distribution_2.png](https://i.imgur.com/Bew6rjz.png)

期望：`Eξ=np`； 方差：`Dξ=npq`； 其中`q=1-p`

[bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution "bernoulli distribution")