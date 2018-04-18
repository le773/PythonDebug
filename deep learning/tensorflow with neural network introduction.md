### 1.0 初识别tensorflow
一个 "`TensorFlow Session`" 是用来运行图的环境。这个 `session` 负责分配 `GPU(s)` `/` 或 `CPU(s)`，包括远程计算机的运算。

张量：简单理解为多维数组。

张量在tensorflow中实现并不是直接采用数组的形式，它只是**对tensorflow的运算结果的引用**。

### 2.0 ReLU和Softmax激活函数
#### 2.1 sigmoid激活函数

![sigmoid_2.png](https://i.imgur.com/NSLxA2Y.png)

正如在反向传播资料中提到的，`S`型函数的导数最大值为 `0.25`（如上所示）。这意味着，当你用`S`型函数单元进行反向传播时，网络上每层出现的错误至少减少 `75%`，如果有很多层，权重更新将很小，这些**权重需要很长的训练时间**。因此，`S`型函数不适合作为隐藏单元上的激活函数。

#### 2.2 初识修正线性单元(ReLU)
##### 2.2.1 修正线性单元(ReLU)
`f(x)=max(x,0)`

![ReLU](https://i.imgur.com/uKvxIEq.png)

当输入是正数时，导数是1，所以没有`S`型函数的反向传播错误导致的消失效果。研究表明，对于大型神经网络来说，`ReLU`的训练速度要快很多。`TensorFlow`和`TFLearn`等大部分框架使你能够轻松地在隐藏层使用`ReLU`，你不需要自己去实现这些`ReLU`。

**遗憾的是**，`ReLU` 单元在训练期间可能会很脆弱并且会变得“无效”。例如，流经 `ReLU` 神经元的大型梯度可能会导致权重按以下方式更新：神经元将再也不会在任何数据点上激活。如果发生这种情况，那么流经该单元的梯度将自此始终为零。也就是说，**ReLU 单元会在训练期间变得无效并且不可逆转**，因为它们可能会不再位于数据流形上。例如，学习速度(`learning rate`)设置的太高，你的网络可能有高达 40% 的神经元处于“无效”状态（即神经元在整个训练数据集上从未激活）。如果能正确地设置学习速度，那么该问题就不太容易出现。

##### 2.2.2 带泄露ReLU
![Leaky_ReLU_1.png](https://i.imgur.com/czJEYs4.png)

当`z<0`时，导数不为`0`，有一个很平缓的斜率

`ReLU`的优点，对很多`z`空间，激活函数的导数和`0`差很远(没有像`sigmoid`和`tanh`那样接近于`0`)，所以神经网络的学习速度通常会快很多。
#### 2.3 Softmax
`softmax`函数的数学表达式如下所示，其中`z`是输出层的输入向量（如果你有`10`个输出单元，则`z`中有`10`个元素）。同样，`j`表示输出单元的索引。

![softmax_1.png](https://i.imgur.com/540fjVy.png)

![softmax](https://pic1.zhimg.com/v2-998ddf16795db98b980443db952731c2_r.jpg)

适用于二分类问题(不然绝不使用)，其它情况选择双曲正切函数(`tanh<ReLU`)更合适

##### 2.3.1 代码实现
```
import tensorflow as tf

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax =  tf.nn.softmax(logits)   
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits:logit_data})

    return output
```

#### 2.4 双曲正切函数 tanh
公式如下：

![tanh](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D177/sign=adf59bb1f51fbe09185ec7135c610c30/96dda144ad345982648941550bf431adcaef84f2.jpg)

图形如下：

![tanh图形](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike92%2C5%2C5%2C92%2C30/sign=4a65cdbb4b34970a537e187df4a3baad/29381f30e924b8994bb77cac64061d950b7bf69f.jpg)

`tanh`几乎任何场合总比`sigmoid`函数效果更好(除二分类外)，因为现在函数输出介于`-1`和`1`之间激活函数的平均值，就更接近于`0`。(类似数据中心化的效果)

**sigmoid tanh缺陷**，如果`z`非常大或非常小，那么导数的梯度或这个函数的斜率可能就很小，这样会**拖慢梯度下降**算法。

#### 2.5 双曲正切函数的导数
双曲正切函数的导数是双曲余弦的平方的倒数。

![tanh_derivative_1](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D124/sign=7bf40c48dec451daf2f608e982fd52a5/1f178a82b9014a909622d2abaa773912b31bee8c.jpg)

#### 2.6  如何选择激活函数
- 如果输出值是`0`和`1`(二分类问题)，那么`sigmoid`函数很适合作为输出层的激活函数
- 然后其它所有单元都用`ReLU`

#### 2.7 为什么需要非线性激活函数
如果没有非线性激活函数，不管神经网络有多少层，一直在做的只是计算线性激活函数，所以不如直接去掉全部隐藏层。

非线性激活函数可以压缩数据(中心化等)

线性激活函数在隐藏层使用可能是在一个需要实数的网络，比如预测房价。
线性函数通常用在输出层。

### 3.0 TensorFlow 交叉熵


### 4.0 随机梯度下降法
![随机梯度下降_1.png](https://i.imgur.com/Je29fc6.png)

解决批量梯度下降运算慢：**随机**从数据集中抽取的很小一部分的平均损失。

如果样本不够随机，那它就完全不再有效；因此，将取出数据集中非常小的一片，计算那些样本的损失和导数，并假设那个导数就是进行梯度下降正确的方向，它并不是每次都是正确的方向，实际上它偶尔还会增加实际的损失，而不是减少它，但我们通过每次执行非常小的步幅，多次执行这个过程来补偿它，因此每一步非常容易计算，但也付出了代价：相比于一大步，需要走很多小步；总的来说，相比于批量梯度下降，这异常有效。因此随机梯度下降在数据和模型尺寸方面扩展很好，同时大数据就大模型。由于它本质上是一个非常差的优化器，碰巧它又是唯一足够快的，实际中能解决很多问题。

### 5.0 Momentum 与学习率衰减
归一化`(u,σ)`对`SGD`(随机梯度)很重要，用方差较小的随机权重也是一样的道理

- 动量

充分利用先前经历的步长，来判断找极小值的方向。
方法：追踪梯度的实时平均值，用该值代替当前一批数据计算得出的方向。

![随机梯度下降_2.png](https://i.imgur.com/Lej3ZHf.png)

- 学习衰减率

### 6.0 参数空间
学习曲率与学习速度

![learning_rate_runing_1.png](https://i.imgur.com/bTXDphB.png)
```
学习速度与学习效果没有必然的联系
```
```
Hyper-parameters
- initial learning rate
- learning rete decay
- momentum
- batch size
- weight initialization
```
如果毫无进展，首先尝试降低学习率。

adagrada:sgd的改进，自动选择`momentum`、`learning rate`，学习过程对超参数不敏感。

### 7.0 mini-batch
`Mini-batching` 是一个一次训练数据集的一小部分，而不是整个训练集的技术。它可以使内存较小、不能同时训练整个数据集的电脑也可以训练模型。

`Mini-batching` 从运算角度来说是低效的，因为你不能在所有样本中计算`loss`。但是这点小代价也比根本不能运行模型要划算。

它跟随机梯度下降（`SGD`）结合在一起用也很有帮助。方法是在每一代训练之前，对数据进行随机混洗，然后创建`mini-batches`，对每一个 `mini-batch`，用梯度下降训练网络权重。因为这些`batches`是随机的，你其实是在对每个`batch`做随机梯度下降（`SGD`）。

### 8.0 Epochs（代）
一个`epoch`（代）是指整个数据集正向反向训练一次。它被用来提示模型的准确率并且不需要额外数据。

### 9.0 保存和读取TensorFlow
 

### 10.0 参数微调
很多时候你想调整，或者说“微调”一个你已经训练并保存了的模型。但是，把保存的变量直接加载到已经修改过的模型会产生错误。

```
InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match.
```
解决：`tf.Variable`指定`name`;

### 11.0 正则化
- 训练停止

![训练停止_1.png](https://i.imgur.com/Y2FACht.png)

- 正则化
隐式的减少自由参数的变量，同时不会使其变得更难优化

深度学习正则化：L2范式(惩罚权重大的项,L2代表各个元素的平方和)

![L2_Regularization_1.png](https://i.imgur.com/xXNVrok.png)

### 12.0 Dropout
从一层到下一层的值通常称为激活。

![dropout_1.png](https://i.imgur.com/PqFTpOb.png)

Dropout 是一个**降低过拟合**的正则化技术。它在网络中暂时的丢弃一些单元（神经元），以及与它们的前后相连的所有节点。

![dropout_4.png](https://i.imgur.com/Btbw8UI.jpg)

![dropout_2.png](https://i.imgur.com/cMy9Xkm.png)

对于训练网络的每个样本，随机将某些的一半设为0，这样，基本上随机地将流经网络的一半数据完全摧毁，然后再一次随机这么做。目的是，**神经网络将永不依赖于任何给定的激活**去存在，因为它们可能随时被摧毁，所以它被迫学习一切的冗余表示，以确保至少将一些信息保存下来。

神经网络通过学习冗余，在实践中，使网络**更加稳固**，并能**防止过拟合**，它也使网络如同在网络集合中达成共识。

### 13.0 Dropout
![dropout_3.png](https://i.imgur.com/dUY5fD3.png)

在训练过程中，不仅对于丢掉的激活值使用0代替，对其他激活值放大到两倍，在这种方法下，在**评估时对模型取平均值时，删除丢弃值，并且合理缩放其他值**即可，所以结果是，将得到一个被合理缩放，激活的平均值。

`tf.nn.dropout()`函数有两个参数：
```
hidden_layer：你要应用 dropout 的 tensor
keep_prob：任何一个给定单元的留存率（没有被丢弃的单元）
```
`keep_prob`可以让你调整丢弃单元的数量。为了补偿被丢弃的单元，`tf.nn.dropout()`把所有保留下来的单元（没有被丢弃的单元）* 1/`keep_prob`

只有在训练模型时我们会丢弃单元，一个好的`keep_prob`初始值是0.5。

在验证或测试时，把 `keep_prob` 值设为1.0 ，这样**保留所有的单元，最大化模型的能力**。

##### `dropout`实例
```
# Solution is available in the other "solution.py" tab
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))
```
