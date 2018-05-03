## 1.0 逻辑回归和SVM的区别是什么？
### 1.1 逻辑回归和SVM的相同点
1. LR SVM都是分类算法
LR的Label是离散的，归于回归算法并不合理。

1. 原始的LR和SVM都是线性分类器

1. LR和SVM都是监督学习算法。

1. LR和SVM都是判别模型。

判别模型会生成一个表示P(Y|X)的判别函数（或预测模型），而生成模型先计算联合概率p(Y,X)然后通过贝叶斯公式转化为条件概率。简单来说，在计算判别模型时，不会计算联合概率，而在计算生成模型时，必须先计算联合概率。

常见的判别模型有：KNN、SVM、LR，

常见的生成模型有：朴素贝叶斯，隐马尔可夫模型。

----------

### 1.2 逻辑回归和SVM不同点
#### 第一，SVM和正则化的逻辑回归的损失函数

逻辑回归的损失函数：

![logistic_cost_16.png](https://i.imgur.com/ByRAjGb.png)

支持向量机的损失函数：

![svm_cost_1.png](https://i.imgur.com/NNE80br.png)

逻辑回归采用对数损失函数，SVM采用hinge(合页)损失。

SVM:Loss(z) = max(0, 1-z)

LR :Loss(z) = log(1+exp(-z))

其中，g(z) = (1+exp(-z))<sup>-1</sup>


![9345794f944ea3de0f18f02867b86d3f_hd.jpg](https://i.imgur.com/o1M2acp.jpg)

其实，这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。

SVM的处理方法是只考虑 support vectors，也就是和分类最相关的少数点，去学习分类器。

逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重,

两者的根本目的都是一样的。

#### 第二，支持向量机只考虑局部的边界线附近的点，而逻辑回归考虑全局
#### 第三，在解决非线性问题时，支持向量机采用核函数的机制，而LR通常不采用核函数的方法。
假设我们在LR里也运用核函数的原理，那么每个样本点都必须参与核计算，这带来的计算复杂度是相当高的。

#### 第四，​线性SVM依赖数据表达的距离测度，所以需要对数据先做normalization，LR不受其影响。

#### 第五，SVM的损失函数就自带正则
![svm_cost_2.png](https://i.imgur.com/E1zTwg6.png)

#### 其他的
1. 损失函数优化方法不同。逻辑回归用剃度下降法优化，svm用smo方法进行优化；
2. 处理数据规模不同。LR一般用来处理大规模的学习问题，如十亿级别的样本，亿级别的特征。
3. LR本身就是基于概率的，所以它产生的结果代表了分成某一类的概率，而SVM则因为优化的目标不含有概率因素，所以其不能直接产生概率。
4. 因为SVM是基于距离的，而LR是基于概率的，所以LR是不受数据不同维度测度不同的影响，而SVM因为要最小化0.5*||w||<sup>2</sup>所以其依赖于不同维度测度的不同，如果差别较大需要做normalization，当然如果LR要加上正则化时，也是需要normalization一下的。 

参考：

1. [LR与SVM的异同](https://www.cnblogs.com/zhizhan/p/5038747.html)

2. [Linear SVM 和 LR 有什么异同？](https://www.zhihu.com/question/26768865/answer/34078149)

3. [Linear SVM 和 LR 的联系和区别](https://blog.csdn.net/haolexiao/article/details/70191667)

4. [机器学习算法比较——LR vs. SVM](http://www.libinx.com/2017/2017-03-09-comparison-of-algorithm-LR-vs-SVM/)