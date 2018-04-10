### 1.0 初识别tensorflow
一个 "`TensorFlow Session`" 是用来运行图的环境。这个 `session` 负责分配 `GPU(s)` `/` 或 `CPU(s)`，包括远程计算机的运算。

张量：简单理解为多维数组。
张量在tensorflow中实现并不是直接采用数组的形式，它只是对tensorflow的运算结果的引用。

### 2.0 ReLU和Softmax激活函数
#### 2.1 sigmoid激活函数

![sigmoid_2.png](https://i.imgur.com/NSLxA2Y.png)

正如在反向传播资料中提到的，S 型函数的导数最大值为 0.25（如上所示）。这意味着，当你用 S 型函数单元进行反向传播时，网络上每层出现的错误至少减少 75%，如果有很多层，权重更新将很小，这些**权重需要很长的训练时间**。因此，S 型函数不适合作为隐藏单元上的激活函数。

#### 2.2 初识修正线性单元（ReLU）
`f(x)=max(x,0)`

![ReLU](https://s3.cn-north-1.amazonaws.com.cn/u-img/1e33c195-9796-4d18-8752-bc956f5ddc10)

当输入是正数时，导数是1，所以没有`S`型函数的反向传播错误导致的消失效果。研究表明，对于大型神经网络来说，`ReLU`的训练速度要快很多。`TensorFlow`和`TFLearn`等大部分框架使你能够轻松地在隐藏层使用`ReLU`，你不需要自己去实现这些`ReLU`。

**遗憾的是**，ReLU 单元在训练期间可能会很脆弱并且会变得“无效”。例如，流经 ReLU 神经元的大型梯度可能会导致权重按以下方式更新：神经元将再也不会在任何数据点上激活。如果发生这种情况，那么流经该单元的梯度将自此始终为零。也就是说，**ReLU 单元会在训练期间变得无效并且不可逆转**，因为它们可能会不再位于数据流形上。例如，学习速度（learning rate）设置的太高，你的网络可能有高达 40% 的神经元处于“无效”状态（即神经元在整个训练数据集上从未激活）。如果能正确地设置学习速度，那么该问题就不太容易出现。

#### 2.3 Softmax
`softmax`函数的数学表达式如下所示，其中`z`是输出层的输入向量（如果你有`10`个输出单元，则`z`中有`10`个元素）。同样，`j`表示输出单元的索引。
![softmax_1.png](https://i.imgur.com/540fjVy.png)

![softmax](https://pic1.zhimg.com/v2-998ddf16795db98b980443db952731c2_r.jpg)

##### 代码实现
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

### 3.0 TensorFlow 交叉熵


### 4.0 随机梯度下降法
![随机梯度下降_1.png](https://i.imgur.com/Je29fc6.png)

解决批量梯度下降运算慢：**随机**从数据集中抽取的很小一部分的平均损失。

如果样本不够随机，那它就完全不再有效；因此，将取出数据集中非常小的一片，计算那些样本的损失和导数，并假设那个导数就是进行梯度下降正确的方向，它并不是每次都是正确的方向，实际上它偶尔还会增加实际的损失，而不是减少它，但我们通过每次执行非常小的步幅，多次执行这个过程来补偿它，因此每一步非常容易计算，但也付出了代价：相比于一大步，需要走很多小步；总的来说，相比于批量梯度下降，这异常有效。因此随机梯度下降在数据和模型尺寸方面扩展很好，同时大数据就大模型。由于它本质上是一个非常差的优化器，碰巧它又是唯一足够快的，实际中能解决很多问题。

### 5.0 Momentum 与学习率衰减
归一化(u,σ)对SGD(随机梯度)很重要，用方差较小的随机权重也是一样的道理

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

### mini-batch
Mini-batching 是一个一次训练数据集的一小部分，而不是整个训练集的技术。它可以使内存较小、不能同时训练整个数据集的电脑也可以训练模型。

Mini-batching 从运算角度来说是低效的，因为你不能在所有样本中计算 loss。但是这点小代价也比根本不能运行模型要划算。

它跟随机梯度下降（SGD）结合在一起用也很有帮助。方法是在每一代训练之前，对数据进行随机混洗，然后创建 mini-batches，对每一个 mini-batch，用梯度下降训练网络权重。因为这些 batches 是随机的，你其实是在对每个 batch 做随机梯度下降（SGD）。

### Epochs（代）
一个 epoch（代）是指整个数据集正向反向训练一次。它被用来提示模型的准确率并且不需要额外数据。


