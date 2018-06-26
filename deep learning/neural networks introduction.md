## 神经网络入门
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
#### 4.4.1 mini-batch梯度下降

mini-batch梯度下降：在每次更新时用b个样本,其实批量的梯度下降就是一种折中的方法，他用了一些小样本来近似全部的，其本质就是我1个指不定不太准，那我用个30个50个样本那比随机的要准不少了吧，而且批量的话还是非常可以反映样本的一个分布情况的。在深度学习中，这种方法用的是最多的，因为这个方法收敛也不会很慢，收敛的局部最优也是更多的可以接受！

代价函数

![gd_costfunction_3.jpg](https://i.imgur.com/uVKFAr1.jpg)

参考
[三种梯度下降的方式：批量梯度下降、小批量梯度下降、随机梯度下降](https://blog.csdn.net/uestc_c2_403/article/details/74910107 "三种梯度下降的方式：批量梯度下降、小批量梯度下降、随机梯度下降")

[如何理解随机梯度下降(Stochastic gradient descent，SGD)？](https://www.zhihu.com/question/264189719/answer/291167114 "如何理解随机梯度下降(Stochastic gradient descent，SGD)？")

梯度下降引向局部最低点

![gradient_descent_1.png](https://i.imgur.com/YJn6nyu.png)

##### 4.4.2 Mini-batch gradient descent size
在合理范围内，增大 Batch_Size 有何好处？

- 内存利用率提高了，大矩阵乘法的并行化效率提高。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
- 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。

盲目增大 Batch_Size 有何坏处？
- 内存利用率提高了，但是内存容量可能撑不住了。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
- Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

参考：[深度机器学习中的batch的大小对学习效果有何影响？](https://www.zhihu.com/question/32673260 "深度机器学习中的batch的大小对学习效果有何影响？")

##### 4.4.3 不同学习率导致不同的梯度下降
![learningrates.jpeg](https://i.imgur.com/fdoiZkr.jpg)

#### 4.5 Gradient descent with momentum动量梯度下降法
**基本思想**：计算梯度的指数加权平均数，并利用该梯度更新权重

![gd_exponentially_weighted_average_1.png](https://i.imgur.com/ZdLcIUk.png)

上图，蓝色为普通梯段下降，红色为momentum梯度下降

动量梯度下降**目的**：纵向上减小摆动，横向上加快学习速度。

![gd_exponentially_weighted_average_2.png](https://i.imgur.com/u37Sw4X.png)

**关于偏差修正**：因为10次迭代后，移动平均已经过了初始阶段，不再是一个具有偏差的预测，所以`dw`、`db`不在受到偏差修正的困扰

```python
def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))

  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  v = config['momentum'] * v - config['learning_rate'] * dw
  next_w = w + v
  #############################################################################
  config['velocity'] = v

  return next_w, config
```

#### 4.6 RMSprop 加快梯度下降
![RMSprop_1.png](https://i.imgur.com/NEIRrP9.png)

`db`较大，`dw`较小，所以纵轴消除摆动，横轴加快速度。更大的`α`可以加快此速率。

**核心算法**:

![RMSprop_core.png](https://i.imgur.com/QKLvWsE.png)

```python
def rmsprop(x, dx, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  next_x = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #
  # config['cache'].                                                          #
  #############################################################################
  config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx ** 2
  next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
  #############################################################################

  return next_x, config
```
#### 4.7 Adam(Adaptive Moment Estimation) 自适应矩估计
![Adam_1.png](https://i.imgur.com/Nxq1wvA.png)

`Adam`是`momentum`和`RMSpro`p的结合，`β1`是第一阶矩，一般`0.9`，`β2`是第二阶矩，一般`0.999`，`ϵ`一般`10−8`。

```python
def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)

  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2)
  next_x = x - config['learning_rate'] * config['m'] / (np.sqrt(config['v']) + config['epsilon'])
  #############################################################################

  return next_x, config
```
#### 4.8 学习衰减率
学习衰减率的计算方法：

![learning_attenuation_1.png](https://i.imgur.com/oOvVIYt.png)

`decay_rate`:衰减率
`epoch_num`:所有的训练样本完整训练一遍的次数。

#### 4.9 几种优化方式的区别
![opt2.gif](https://i.imgur.com/yKXIthQ.gif)

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

#### 5.3 梯度消失与梯度下降
![gd_exploding_1.png](https://i.imgur.com/Dv7uXuX.png)

在神经网络中，与层数`L`相关的导数或梯度下降、，就是呈现指数增长或指数增长下降。

如果作为`L`的函数的激活函数或梯度函数以指数级增长或递减，它们的值将变得很大，从而导致训练难度上升，尤其是梯度与`L`相差指数级，梯度下降算法的步长 会非常非常小，将会花费很长时间来学习

- `ReLU`
梯度消失和梯度爆炸，在`relu`下都存在，随着网络层数变深，`activations`倾向于越大和越小的方向前进，往大走梯度爆炸（回想在求梯度时，每反向传播一层，都要乘以这一层的`activations`），往小走进入死区，梯度消失。 这两个问题最大的影响是，**深层网络难于converge**。

- `sigmoid`
`sigmoid`不存在梯度爆炸，在`activations`往越大越小的方向上前进时，梯度变化太大，梯度都会消失。

参考：[怎么理解梯度弥散和梯度爆炸呢？](https://www.zhihu.com/question/66027838/answer/237409864)

#### 5.4 神经网络的权重初始化
![neural network init 1](https://i.imgur.com/WYDxurh.png)

- 对于`ReLU`激活函数选择 1
![weightinit_relu.png](https://i.imgur.com/o1VXcN3.png)
```
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)
```
- 对于`Tanh`激活函数选择 2
![weightinit_tanh.png](https://i.imgur.com/L0fMJDf.png)
```
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
```
- 其它
![weightinit_other.png](https://i.imgur.com/4OQoLAY.png)
```
W = np.random.randn(fan_in, fan_out) / np.sqrt((fan_in + fan_out)/2)
```
#### 5.5 梯度的数值逼近
单边误差

![neural_network_error_1.png](https://i.imgur.com/8pH7iXe.png)

双边误差

![neural_network_error_2.png](https://i.imgur.com/pYo1TWg.png)

由上对比可知，双边误差更接近与`f(θ)`的导数

#### 5.6 梯度校验
![gradient_checking_1.png](https://i.imgur.com/FKb51Vp.png)

#### 5.7 关于梯度校验实现的建议
![gradient_checking_2.png](https://i.imgur.com/ygHY9kt.png)

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


#### 7.1 参考代码
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

### 9.0 为什么神经网络在考虑梯度下降的时候，网络参数的初始值不能设定为全0，而是要采用随机初始化思想？

![neural_net_param_init_1.png](https://i.imgur.com/U3v1rDQ.png)

如果初始化权重相同，所有的隐藏单元都是对称的，隐藏单元都在计算相同的函数得到相同的值，然后不停的迭代，不停的相同，不停的迭代，不停的相同......，最后就得到了相同的值（权重和截距），所以永远不可能找到最优值。

![neural_net_param_init_2.png](https://i.imgur.com/vmM83h2.png)

通常把神经网络初始化成非常非常小的随机值，如果初始化很大，那么激活函数的值(接近饱和)可能落在比较平缓处，梯度比较小，意味着梯度下降法会非常慢。

参考 [为什么神经网络在考虑梯度下降的时候，网络参数的初始值不能设定为全0，而是要采用随机初始化思想？](https://www.zhihu.com/question/36068411/answer/65751656 "为什么神经网络在考虑梯度下降的时候，网络参数的初始值不能设定为全0，而是要采用随机初始化思想？")

### 9.1 为什么使用深层表示？
深度学习中的电路理论

![neural_net_8.png](https://i.imgur.com/qpHQi3x.png)

深层表示每一层的隐藏单元可以很小；如果神经网络深度很浅，那么需要的隐藏单元呈现指数增长`(2**(n-1))`。

### 9.2 参数 & 超参数
学习曲率`α`、隐藏层数、`momentum`、`mini-batch`、正则化参数等能控制`w`、`b`的参数，称为超参数。某种程度能决定`w`和`b`。

`w`和`b`，即为参数。
### 9.3 参数调优
#### 9.3.1 参数调优
![Coarse_to_fine_1.png](https://i.imgur.com/I1btnGp.png)

先在整个区域随机的选取参数，找到较优的区间，然后在这个区间更密集的取点。通过实验超参数的不同取值，可以选择对于训练目标而言的最优值。
#### 9.3.2 各种超参数对模型容量的影响
![hyper_parameters_1.png](https://i.imgur.com/xbRuNSX.png)

### 9.4 Appropriate scale for hyperparameters
![Appropriate_scale_for_hyperparameters_1.png](https://i.imgur.com/BnPCqFI.png)

随机选参数，不是随机均匀的选择，而是在合理的标尺上选择。上图是学习衰减率`α`的实例，`β`同理。

### 10 Q&A
#### 10.1 神经网络隐藏层节点的个数过多、不足会怎样?
若隐层节点数太少，网络可能根本不能训练或网络性能很差；若隐层节点数太多，虽然可使网络的系统误差减小，但一方面使网络训练时间延长；</br>
另一方面，训练容易陷入局部极小点而得不到最优点，也是训练时出现“过拟合”的内在原因。</br>
因此，合理隐层节点数应在综合考虑网络结构复杂程度和误差大小的情况下用节点删除法和扩张法确定。

参考：
1.[神经网络权重初始化问题](https://blog.csdn.net/marsggbo/article/details/77771497)
