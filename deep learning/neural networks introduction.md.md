##　神经网络入门
### 1.0 neural networks
![neuralnetwords_1.png](https://i.imgur.com/VWldwr3.png)

- 第一层:代表两项成绩的输入端
- 第二层:根据两项成绩对应的坐标，检查数据点是否越过了两条直线
- 第三层:根据前程取得的输出结果对其进行‘与运算’

### 2.0 Perceptron 感知器
数据，无论是考试成绩还是评级，被输入到一个相互连接的节点网络中。这些独立的节点被称作**感知器** 或者**神经元**。它们是构成神经网络的基本单元。每个感知器依照输入数据来决定如何对数据分类。

权重:当数据被输入感知器，它会与分配给这个特定输入的权重相乘。权重决定神经元对决定的重要性。

训练：根据之前权重下分类的错误来调整权重。

![perceptron_1.png](https://i.imgur.com/5ysBs8t.jpg)

感知器求和的结果会被转换成输出信号，这是通过把线性组合传给**激活函数**来实现的。

最简单的激活函数：单位阶跃函数

![Heaviside_step_function_1.png](https://i.imgur.com/IdIA7jN.gif)

感知器公式:

![感知器公式.png](https://i.imgur.com/9AQZfrh.gif)

### 3.0 最简单的神经网络
![neuralnetwords_2.png](https://i.imgur.com/GGP0V0k.png)

神经网络示意图，圆圈代表单元，方块是运算。
求和：`y=∑wi*x + b`

![sigmoid_1.png](https://i.imgur.com/Fg8mEIk.png)

`sigmoid(x)=1/(1+exp(−y))`

### 4.0 梯度下降
#### 4.1.1 梯度下降实用技巧
##### 4.1.1.1 梯度下降实用技巧I之特征缩放
当多个特征的范围差距过大时，代价函数的轮廓图会非常的偏斜，这会导致梯度下降函数收敛的非常慢。因此需要特征缩放(feature scaling)来解决这个问题，特征缩放的目的是把特征的范围缩放到接近的范围。当把特征的范围缩放到接近的范围，就会使偏斜的不那么严重。通过代价函数执行梯度下降算法时速度回加快，更快的收敛。

##### 4.1.1.2 梯度下降实用技巧II之学习率
下降的幅度过大，跳过了全局最小值（下图下方所示的图形问题），解决办法也是缩小学习率α的值

#### 4.2 批量梯度下降法

批量梯度下降法（Batch Gradient Descent，简称BGD）是梯度下降法最原始的形式，它的具体思路是在更新每一参数时都使用所有的样本来进行更新，也就是方程（1）中的m表示样本的所有个数。

- 优点：全局最优解；易于并行实现；
- 缺点：当样本数目很多时，训练过程会很慢。

代价函数

![gd_costfunction_1.jpg](https://i.imgur.com/lH4709M.jpg)

#### 4.3 随机梯度下降法

随机梯度下降法：它的具体思路是在更新每一参数时都使用一个样本来进行更新，也就是方程（1）中的m等于1。每一次跟新参数都用一个样本，更新很多次。如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次，这种跟新方式计算复杂度太高。
但是，SGD伴随的一个问题是噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。

- 优点：训练速度快；
- 缺点：准确度下降，并不是全局最优；不易于并行实现。

代价函数

![gd_costfunction_2.jpg](https://i.imgur.com/4t2yo3r.jpg)

#### 4.4 mini-batch梯度下降

mini-batch梯度下降：在每次更新时用b个样本,其实批量的梯度下降就是一种折中的方法，他用了一些小样本来近似全部的，其本质就是我1个指不定不太准，那我用个30个50个样本那比随机的要准不少了吧，而且批量的话还是非常可以反映样本的一个分布情况的。在深度学习中，这种方法用的是最多的，因为这个方法收敛也不会很慢，收敛的局部最优也是更多的可以接受！

代价函数

![gd_costfunction_3.jpg](https://i.imgur.com/uVKFAr1.jpg)

参考
[三种梯度下降的方式：批量梯度下降、小批量梯度下降、随机梯度下降](https://blog.csdn.net/uestc_c2_403/article/details/74910107 "三种梯度下降的方式：批量梯度下降、小批量梯度下降、随机梯度下降")

[如何理解随机梯度下降(Stochastic gradient descent，SGD)？](https://www.zhihu.com/question/264189719/answer/291167114 "如何理解随机梯度下降(Stochastic gradient descent，SGD)？")

梯度下降引向局部最低点

![gradient_descent_1.png](https://i.imgur.com/YJn6nyu.png)

### 5.0 梯度下降：数学
E = (y - ŷ)**2,(使用平方较小误差的惩罚值较低，较大误差惩罚值较大)

![梯度下降_权重求导_0.png](https://i.imgur.com/iXvkn3C.png)

![梯度下降_权重求导_1.png](https://i.imgur.com/vXXRoLd.png)

![梯度下降_权重求导_2.png](https://i.imgur.com/uXRVy9g.png)

![梯度下降_权重求导_3.png](https://i.imgur.com/u3nKKX1.png)

![梯度下降_权重求导_4.png](https://i.imgur.com/jUKA3ir.png)

#### 5.1 梯度下降来更新权重的算法概述
![gradient_descent_weight_1.png](https://i.imgur.com/hqKuvVZ.png)

也可以对每条记录更新权重，而不是把所有记录都训练过之后再取平均。
使用sigmoid作为激活函数；

代码
```
import numpy as np
from data_prep import features, targets, features_test, targets_test

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# TODO: We haven't provided the sigmoid_prime function like we did in
#       the previous lesson to encourage you to come up with a more
#       efficient solution. If you need a hint, check out the comments
#       in solution.py from the previous lecture.

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Note: We haven't included the h variable from the previous
        #       lesson. You can add it if you want, or you can calculate
        #       the h together with the output

        # TODO: Calculate the output
        output = sigmoid(np.dot(x, weights))

        # TODO: Calculate the error
        error = output - y

        # TODO: Calculate the error term
        error_term = error * output * (1 - output)

        # TODO: Calculate the change in weights for this sample
        #       and add it to the total weight change
        del_w += learnrate * np.dot(error_term, x)

    # TODO: Update weights using the learning rate and the average change in weights
    weights += np.dot(error_term, x)

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```

#### 5.2 梯度下降法是万能的模型训练算法吗？
当然不是，一方面不是所有的代价可汗可导，即使可导也可能面临导数下降慢，计算效率低的情况，
另一方面，不能处理局部最优很多的情况。

下图右一，为包含多个局部最优解

![gd_local_optimal_model_1.jpg](https://i.imgur.com/juO4zT6.jpg)

**一方面**，梯度并不是在任何时候都可以计算的。实际中很多问题的目标函数并不是可导的，这时梯度下降并不适用，这种情况下一般需要利用问题的结构信息进行优化，比如说Proximal gradient方法。甚至有些问题中目标函数的具体形式都不知道，更别谈求梯度，比如说Bayesian Optimization。

**另一方面**，即使问题可导，梯度下降有时并不是最佳选择。梯度下降的性能跟问题的条件数相关，在条件数比较大时问题中梯度下降可能会非常慢。相对来说，以拟牛顿法为代表的二阶方法没有这个问题，虽然拟牛顿法在高维问题中会有计算量偏大的问题，但在很多场景还是比梯度下降有优势。再比如，在梯度计算代价比较大时，SGD及其变种会远比普通的梯度下降快。

当然，**总体来说**，在机器学习的各种教科书中梯度下降是最常见的优化方法。主要因为它非常简单易懂，而且大多数情况下效率比较高，但同时也是因为机器学习中大多数问题的惩罚函数是比较smooth的。



参考：[梯度下降法是万能的模型训练算法吗？](https://www.zhihu.com/question/38677354/answer/85769046)


### 6.0 多层感知器
![多层感知器_1.png](https://i.imgur.com/s8bvArC.png)

权重被储存在矩阵中，由w{ij}来索引。矩阵中的每一行对应从同一个输入节点发出的权重，每一列对应传入同一个隐藏节点的权重。

#### 6.1 隐藏层
隐藏层的意义，是把前一层的向量变成新的向量。

也就是坐标变换，也就是把空间扭曲一下变下维度，让他们更加线性可分。

![hidden_layer_1.jpg](https://i.imgur.com/IWEL5y4.jpg)

[神经网络中隐层有确切的含义吗？](https://www.zhihu.com/question/60493121)
#### 6.2 隐藏层的计算公式
![多层感知器_2.png](https://i.imgur.com/BsbtKuz.png)

### 7.0 反向转播
要使用梯度下降法更新隐藏层的权重，你需要知道各隐藏层节点的误差对最终输出的影响。每层的输出是由两层间的权重决定的，两层之间产生的误差，按权重缩放后在网络中向前传播。既然我们知道输出误差，便可以用权重来反向传播到隐藏层。

**例如**，输出层每个输出节点 k的误差是![多层感知器_3.png](https://i.imgur.com/CmJK5eu.png)，隐藏节点j的误差即为输出误差乘以输出层-隐藏层间的权重矩阵（以及梯度）。

![反向传播_1.png](https://i.imgur.com/BUwPI21.gif)

注释：j到k间映射的权重 `x` k层的误差项 `x` k层预测点导数值

```
(y-ŷ) = 上层的误差项 `x` 本层的输出
```

**然后**，梯度下降与之前相同，只是用新的误差(反向传播误差的权重更新步长)：

![反向传播_2.png](https://i.imgur.com/Ua18QeS.gif)

注释：j到k间映射的权重 `x` k层的误差项 `x` j层的值.


参考代码
```
import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer
hidden_error_term =np.dot(output_error_term, weights_hidden_output) * hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
```

### 8.0 初识别tensorflow
一个 "`TensorFlow Session`" 是用来运行图的环境。这个 `session` 负责分配 `GPU(s)` `/` 或 `CPU(s)`，包括远程计算机的运算。
