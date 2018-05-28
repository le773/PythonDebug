## 特征工程

![feature_1.jpg](https://i.imgur.com/poevApe.jpg)

## 1.0 特征缩放
### 1.0 反正切函数arctan
![artan_1.jpg](https://i.imgur.com/Lobk1EL.jpg)

```
f(x) = arctan(x)*2/PI
```
###  1.1 调节比例
![调节比例](https://i.imgur.com/GfRSDjV.png)

- 优点：预估输出相对稳定
- 缺点：如果输出特征中有异常值，那么特征缩放就会比较棘手（最大值最小值可能是极端值）

###### 应用
```
from sklearn.preprocessing import MinMaxScaler
import numpy
#这里numpy数组中的是特征，因为此处特征只有一个，所以看起来是这样的
#因为这里应该作为一个浮点数进行运算，所以数字后面要加.
weights = numpy.array([[115.],[140.],[175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
print rescaled_weight
```

### 1.2 标准化
![标准化](https://i.imgur.com/9IDUEHR.png)

特征标准化使每个特征的值有平均值(zero-mean)和单位方差(unit-variance)。

### 1.3 对数log
```
# 使用自然对数缩放数据
log_data = np.log(data)
```
如果数据**不是正态分布**的，尤其是数据的**平均数和中位数相差很大**的时候（表示数据非常歪斜）。这时候通常用一个**非线性**的缩放是很合适的。
尤其是对于金融数据。一种实现这个缩放的方法是使用**Box-Cox** 变换，这个方法能够计算出能够最佳减小数据倾斜的指数变换方法。一个比较简单的并且在大多数情况下都适用的方法是使用自然对数。

----------

### 1.4 Batch Norm
1. 把具有不同尺度的特征映射到同一个坐标系，具有相同的尺度(相似特征分布)
2. 一定程度上消除了噪声、质量不佳等各种原因对模型权值更新的影响。

含有batch-norm的神经网络计算步骤：

![mini-batch-norm_1.png](https://i.imgur.com/wFGzFb4.png)

对于含有m个节点的某一层神经网络，对z进行操作的步骤为:

![mini-batch-norm_2.png](https://i.imgur.com/sfNipn1.png)

其中的`γ`、`β`并不是超参数，而是两个需要学习的参数，神经网络自己去学着使用和修改这两个扩展参数。这样神经网络就能自己慢慢琢磨出前面的标准化操作到底有没有起到优化的作用。如果没有起到作用，就使用`γ`、`β`来抵消一些之前进行过的标准化的操作。

#### 1.4.1 Batch Norm为什么会奏效？
通过归一化所有的输入特征值x，以获得类似范围的值，可以加快学习。`Batch-norm`是类似的道理

当前的获得的经验无法适应新样本、新环境时，便会发生“`Covariate Shift`”现象。 对于一个神经网络，前面权重值的不断变化就会带来后面权重值的不断变化，批标准化减缓了隐藏层权重分布变化的程度。采用批标准化之后，尽管每一层的z还是在不断变化，但是它们的**均值和方差将基本保持不变，限制了在前层的参数更新会影响数值分布的程度，使得后面的数据及数据分布更加稳定，减少了前面层与后面层的耦合**，使得每一层不过多依赖前面的网络层，最终加快整个神经网络的训练。

#### 1.4.2 Convariate Shift
`Convariate Shift`是指训练集的样本数据和目标样本集**分布不一致**时，训练得到的模型无法很好的`Generalization`(泛化)。它是分布不一致假设之下的一个分支问题，也就是指`Sorce Domain`和`Target Domain`的条件概率一致的，但是其边缘概率不同。的确，**对于神经网络的各层输出，在经过了层内操作后，各层输出分布就会与对应的输入信号分布不同，而且差异会随着网络深度增大而加大了，但每一层所指向的Label仍然是不变的**。

#### 1.4.3 正则化是Batch Norm意想不到的效果
![mini-batch-norm_3.png](https://i.imgur.com/ngj1TAq.png)

1. 均值和方差有一点小噪音，因为它是由一小部分数据估计得出的，所以`mini batch norm`后的数据也是有噪音的。 和`dropout`类似，它往每个隐藏层的激活值上增加了噪音，迫使下一层的隐藏单元不过分依赖任何一个隐藏单元，`mini batch norm`获得轻微的正则化效果，如果想获得更大的正则化效果，可以和`dropout`结合使用。

2. 应用较大的`mini-bach`，(噪音减小)可以减少正则化效果。

----------

### 1.5  其他的转换
![经济学转换方式_1.jpg](https://i.imgur.com/PevGIqH.jpg)

### 1.6 Box-Cox
![Box-Cox.jpg](https://i.imgur.com/RYBEGyi.jpg)

#### 没有Box-Cox变换的回归
![before_Box-Cox.jpg](https://i.imgur.com/QiRZGTK.jpg)

#### Box-Cox变换之后的回归
![after_Box-Cox.jpg](https://i.imgur.com/cDEP7y3.jpg)


### 1.7 哪些机器学习算法会受到特征缩放的影响？
概率模型不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率。
比如：决策树。

- SVM(rbf)计算最大距离时就是这种情况。如果我们把某一点增大至其他点的两倍，那么它的数值也会扩大一倍
- K-均值聚类也是。计算各数据点到集群中心的距离
- 线性回归之类
----------

### 2.0 为什么 feature scaling 会使 gradient descent 的收敛更好?
**如果不归一化**，各维特征的跨度差距很大，目标函数就会是“扁”的：

![gd no scale](https://pic4.zhimg.com/8adda8341490329a5ffcfcd9dc808788_r.jpg)

（图中椭圆表示目标函数的等高线，两个坐标轴代表两个特征）
这样，在进行梯度下降的时候，梯度的方向就会偏离最小值的方向，走很多弯路。
(梯度下降只是**向局部最优的趋近**，而且不能保证每步都是趋近)

**如果归一化**了，那么目标函数就“圆”了：

![gd scale](https://pic3.zhimg.com/80/43c33fb1801c3d35f94b06bd2bfd277c_hd.jpg)

看，每一步梯度的方向都基本指向最小值，可以大踏步地前进。

参考 

[为什么 feature scaling 会使 gradient descent 的收敛更好?](https://www.zhihu.com/question/37129350 "为什么 feature scaling 会使 gradient descent 的收敛更好?")

[CNN 入门讲解：什么是标准化(Normalization)？](https://zhuanlan.zhihu.com/p/35597976)

