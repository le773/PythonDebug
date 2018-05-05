### 01 集成学习的简单规则 
先通过某个数据子集进行学习，形成某个规则，然后通过另一个数据子集进行学习，形成不同的规则，接着通过另一个数据子集进行学习，形成第三个规则，接着更多；最后收集所有这些规则，并将他们合并成为复杂的规则。

#### 为什么考虑子集，而不考虑所有数据？
因为如果考虑所有数据的话，就很难在想到这些简单规则。

### 02.01 集成学习算法
挑选子集的时候遵守均匀规则。

### 02.02 集成Boosting
- 如何改变训练数据的权重或概率分布？
AdaBoost算法提高那些被前一轮弱分类器错误分类的样本的权重，而降低那些被正确分类的权重，这样做的好处是在下一轮的的分类过程中错误的分类由于权重加大而受到更大的关注。

- 如何将弱分类器组合为一个强分类器？
AdaBoost采用加权多数表决的方法。具体说来，即加大分类误差率小的弱分类器的权重，使其在表决中起较大作用，减小分类误差率较大的弱分类器的权重，使其在表决中起较小的作用。

```
# classEst 此轮预测的最优结果
# aggClassEst   预测的分类结果值
# 预测的分类结果值，在上一轮结果的基础上，进行加和操作
aggClassEst += alpha*classEst
# sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
# 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
# sign是表决函数
sign(aggClassEst)
```

### 03 AdaBoost简介

![boosting_1.png](https://i.imgur.com/LjcD27T.png)

AdaBoost是adaptive boosting的缩写，其运行过程如下：

训练数据中的每个样本，并赋予一个权重，这些权重构成向量D。一开始，这些权重都初始化程相等值(`D = mat(ones((m, 1))/m)`)；

首先，在训练数据上训练出一个弱分类器并计算该分类器的错误率，

然后，在同一个数据集上再次训练弱分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次错分的权重将会提高。

为了从所有弱分类器中得到最终分类结果，Adaboost为每个分类器都分配一个权重值，这些alpha值是基于每个分类器错误率进行计算的。

其中，错误率的定义为：

`ε = 未正确分类的样本/所有样本`

alpha的计算公式：

`α = 0.5 * ln((1-ε) / ε)`

计算出alpha值之后，可以对权重向量D进行更新，以使得那些正确分类的样本的权重降低而错分样本的权重升高。D的计算方法如下：

如果某个样本被正确分类，那么该样本的权重更改为：

![adaboost_w_1.png](https://i.imgur.com/ZiINTOq.png)

如果某个样本被错误分类，那么该样本的权重更改为：

![adaboost_w_2.png](https://i.imgur.com/ROk6mTl.png)

在计算出D之后，Adaboost又开始进入下一轮迭代。AdaBoost算法会不断的重复训练和调整权重的过程，直到训练错误率为0或者弱分类器的数目达到用户的指定值为止。

对于无法接受带权样本的基学习算法，可通过“重采样(re-sampling)”来处理，即在每一轮学习中，根据样本分布对训练样本进行采样，再用重采样而得到的样本集对基学习器进行训练。一般而言这两种做法没有显著的优劣差别。

需要注意的是，Boosting算法在训练每一轮都要检查当前生成的基学习器是否满足基本条件(检查当前生成的基学习器是否满足基本条件。(在机器学习实战源码中并未检查，规范来说，应该需要检查))，一旦条件不满足，则当前系学习器即被抛弃且学习过程停止。

在此种情形下，初始设置的学习轮数并未达到，可能导致最综集成中只包含很少的基学习器而性能不佳。若采用“重采样(re-sampling)”，则可获得“重启动”机会以避免训练过程中提前停止，即在抛弃不满足条件的当前基学习器之后，可根据当前分布重新对训练样本进行采样，再基于新的采样结果重新训练出基学习器，而从使学习过程达到T轮。

#### 03.01 AdaBoost算法
![AdaBoost算法](https://i.imgur.com/JMf8LgI.png)

α （模型权重）：目的主要是计算每一个分类器实例的权重(加和就是分类结果)
D （样本权重）：的目的是为了计算错误概率： `weightedError = D.T*errArr`，求最佳分类器
样本的权重值：如果一个值误判的几率越小，那么 D 的样本权重越小

#### 03.02 AdaBoost算法 伪代码
```
adaBoostTrainDS # 返回弱分类器集合 分类结果
    迭代次数
        1. buildStump # 最佳单层决策树
            # 循环所有的feature列，将列切分成 若干份，每一段以最左边的点作为分类节点
                for j in range(-1, int(numSteps)+1) # 步数
                     for inequal in ['lt', 'gt']: # go over less than and greater than
                         # 对单层决策树进行简单分类，得到预测的分类值
                # bestStump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少
            return bestStump| dim            表示 feature列
                            | threshVal      表示树的分界值
                            | inequal        表示计算树左右颠倒的错误率的情况
                   weightedError  表示整体结果的错误率
                   bestClasEst    预测的最优结果
        2. # 计算当前迭代最好的分类的alpha权重值
        3. 根据分类正确错误，更改样本权重
        4. # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
           # aggClassEst += alpha*classEst(预测的最优结果)
        5. 计算错误率，如果小于预期，则使用新的样本权重继续迭代
```

### 03.03 不同规模集成及其基学习器所对应的分类边界
![boosting_8.png](https://i.imgur.com/6Bd7OKz.png)

### 04 AdaBoost总结
#### 04.01 AdaBoost的优缺点：

- 优点
1. Adaboost作为分类器时，分类精度很高
1. 泛化错误率低，不容易发生过拟合；
1. 易编码，可以应用在大部分分类器上，无参数调整；
3. 在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。

- 缺点
对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性

#### 04.02 算法组合
Bagging + 决策树 = 随机森林
AdaBoost + 决策树 = 提升树
Gradient Boosting + 决策树 = GBDT

#### 05.01 Bagging、Boosting二者之间的区别
1.样本选择上
- Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
- Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。

2.样例权重
- Bagging：使用均匀取样，每个样例的权重相等。
- Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。

3.预测函数
- Bagging：所有预测函数的权重相等。
- Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。

4.并行计算
- Bagging：各个预测函数可以并行生成。
- Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

#### 05.02 adaboost为什么不容易过拟合呢？
1. 弱分类器非常简单，即使很多融合也不易过拟合，但如果弱分类器太强，则易过拟合；
2. 分类错误率上界随着训练增加而稳定下降

#### 05.03 Boosting 和 Adaboost 的关系和区别是什么?
boosting 是一种将弱分类器转化为强分类器的方法统称，而adaboost是其中的一种，采用了exponential loss function（其实就是用指数的权重），根据不同的loss function还可以有其他算法。

#### 05.04 Bagging与Boosting的区别
1. 取样方式不同:Bagging采用均匀取样，而Boosting根据错误率取样。
1. Bagging的各个预测函数没有权重，而Boosting是有权重的。
1. Bagging的各个预测函数可以并行生成，而Boosing的各个预测函数只能顺序生成。

#### 05.05 为什么说bagging是减少variance，而boosting是减少bias?
~~Boosting 则是迭代算法，每一次迭代都根据上一次迭代的预测结果对样本进行加权，所以随着迭代不断进行，误差会越来越小，所以模型的 bias 会不断降低。~~

~~Bagging 是 Bootstrap Aggregating 的简称，意思就是再取样 (Bootstrap) 然后在每个样本上用强分类器训练出来的模型取平均，强分类器是低偏差，可能高方差，通过取平均值降低模型的 variance。~~

~~以上解答是正确的，但是显得不够深度~~


Bagging对样本重采样，对每一重采样得到的子样本集训练一个模型，最后取平均。由于子样本集的相似性以及使用的是同种模型，因此各模型有近似相等的bias和variance（事实上，各模型的分布也近似相同，但不独立）。由于![boosting_2.png](https://i.imgur.com/9VCacvc.png)所以bagging后的bias和单个子模型的接近，一般来说不能显著降低bias。另一方面，若各子模型独立，则有

![boosting_3.png](https://i.imgur.com/Gj64wU1.png)

此时可以显著降低variance。
若各子模型完全相同，则

![boosting_4.png](https://i.imgur.com/EJGOb0h.png)

此时不会降低variance。

Bagging方法得到的模型具有一定的相关性，属于上面两个极端状况的中间态，因此可以一定程度降低variance。为了进一步降低variance，Random Forest通过随机选取变量子集做拟合的方式de-correlated(降低相关性)了各子模型，是的variance进一步降低。

Boosting从优化角度来看，是用forward-stagewise这种贪心算法去最优化损失函数

![boosting_5.png](https://i.imgur.com/rWIILi2.png)

例如，常见的AdaBoost即等价于用这种方法最小化exponential loss:

L(y,f(x)) = exp(-y * f(x))

所谓forward-stagewise，就是在迭代的第n步，求解新的子模型f(x)及步长a,来最小化
L(y, f<sub>n-1</sub>(x) + a*f(x))，这里f<sub>n-1</sub>(x)是前n-1步得到的子模型的和。因此boosting是在sequential的最小化损失函数，其bias自然逐步下降。但由于是采取这种sequential、adaptive的策略，各子模型之间是强相关的，于是子模型之和并不能显著降低variance。所以说boosting主要还是靠降低bias来提升预测精度。

#### 05.06 Boost采用不同的损失函数

![loss_1.jpg](https://i.imgur.com/N8pJ4Ck.jpg)

参考：

1. [集成学习之Adaboost算法原理小结](https://www.cnblogs.com/pinard/p/6133937.html)

2. [adaboost为什么不容易过拟合呢？](https://www.zhihu.com/question/41047671 "adaboost为什么不容易过拟合呢？")

3. [Boosting 和 Adaboost 的关系和区别是什么?](https://www.zhihu.com/question/37683881)

4. [为什么说bagging是减少variance，而boosting是减少bias?](https://www.zhihu.com/question/26760839)