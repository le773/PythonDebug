### 0.0 为什么引入非线性激励函数？

如果不用激励函数（其实相当于激励函数是f(x) = x），在这种情况下你每一层输出都是上层输入的线性函数，很容易验证，无论你神经网络有多少层，输出都是输入的线性组合，与没有隐藏层效果相当，这种情况就是最原始的感知机（Perceptron）了。

### 1.0 双层神经网络

在网络里面添加一个隐藏层，可以让它构建更复杂的模型。而且，在隐藏层用非线性激活函数可以让它对非线性函数建模。

一个常用的非线性函数叫`ReLU（rectified linear unit）`。`ReLU`函数对所有负的输入，返回 0；所有 x >0 的输入，返回 x。

### 2.0 Sigmoid
![sigmoid.jpeg](https://i.imgur.com/LKwsN6s.jpg)

优点：
1. 在于输出范围有限，所以数据在传递的过程中不容易发散。
2. 输出范围为(0, 1)，所以可以用作输出层，输出表示概率。
3. 求导容易 y=sigmoid(x), y'=y(1-y)

缺点：
1. 会有梯度消失
2. 不是关于原点对称(根据经验，收敛速度将会是非常慢的)
3. 计算exp比较耗时

#### 2.0.1 Sigmoid梯度消失
![sigmoid gradicent loss.png](https://i.imgur.com/wfm0CJg.png)

当神经元的激活在接近0或1处时会饱和：在这些区域，梯度几乎为0。在反向传播的时候，这个（局部）梯度将会与整个损失函数关于该门单元输出的梯度相乘。因此，如果局部梯度非常小，那么相乘的结果也会接近零，这会有效地“杀死”梯度，几乎就有没有信号通过神经元传到权重再到数据了。
### 2.1 tanh
![tanh.jpeg](https://i.imgur.com/lOnDdt4.jpg)

tanh将实数值压缩到[-1,1]之间。和sigmoid神经元一样，它也存在饱和问题，但是和sigmoid神经元不同的是，它的输出是零中心的。</br>
注意，tanh神经元是一个简单放大的sigmoid神经元，具体说来就是：tanh(x)=2σ(2x)-1

优点：
1. 解决了原点对称问题
2. 比sigmoid更快

缺点：
1. 梯度消失没解决，将导致数据多样化丢失。

### 2.2 ReLU
![relu.jpeg](https://i.imgur.com/NuAkzCG.jpg)

优点：</br>
1. 解决部分梯度消失的问题
2. 对于随机梯度下降的收敛有巨大的加速作用，据称这是由它的线性，非饱和的公式导致的。
3. 相比于sigmoid、tanh神经元含有指数运算等耗费计算资源的操作，而ReLU可以简单地通过对一个矩阵进行阈值计算得到。

缺点：</br>
梯度消失没完全解决(Leaky ReLU，在坐标轴左侧的函数导数很小，解决了这个问题)</br>

##### 2.2.1 ReLU的反向传播
![relubp_1.png](https://i.imgur.com/qAiMmet.png)

### 2.3 Leaky ReLU
![LeakyReLU.png](https://i.imgur.com/676Ah6I.png)

f(x) = max(αx, x)

### 2.4 ELU(Exponential Linear Units)
![ELU.png](https://i.imgur.com/JJ0aX5y.png)

优点：
1. All benefits of ReLU：具有ReLU的所有优点
2. Does not die:神经元不会死亡
3. Closer to zero mean outputs:接近零均值输出

缺点：
1. computation requires exp:需要计算指数

### 2.5 maxout
maxout函数：

![maxout.png](https://i.imgur.com/ob2J3lu.png)

![maxout2.png](https://i.imgur.com/vnTMIbB.png)

第i层有3个节点，红点表示，而第（i+1）层有4个结点，用彩色点表示，此时在第（i+1）层采用maxout（k=3）。我们看到第（i+1）层的每个节点的激活值都有3个值，3次计算的最大值才是对应点的最终激活值。

优点：
1. does not have the basic form of dot product-> nonlinearity
2. Generalizes ReLU and Leaky ReLU:ReLU、Leaky ReLU的泛化
3. Linear Regime! Does not saturate! Does not die!：线性操作和不饱和

缺点：
1. 每个神经元的参数数量增加了一倍，这就导致整体参数的数量激增。

##### 2.2.2 带泄露ReLU
![Leaky_ReLU_1.png](https://i.imgur.com/czJEYs4.png)

当`z<0`时，导数不为`0`，有一个很平缓的斜率

`ReLU`的优点，对很多`z`空间，激活函数的导数和`0`差很远(没有像`sigmoid`和`tanh`那样接近于`0`)，所以神经网络的学习速度通常会快很多。
### 2.7 Softmax
`softmax`函数的数学表达式如下所示，其中`z`是输出层的输入向量（如果你有`10`个输出单元，则`z`中有`10`个元素）。同样，`j`表示输出单元的索引。

![softmax_1.png](https://i.imgur.com/540fjVy.png)

![softmax](https://pic1.zhimg.com/v2-998ddf16795db98b980443db952731c2_r.jpg)

适用于二分类问题(不然绝不使用)，其它情况选择双曲正切函数(`tanh<ReLU`)更合适

##### 2.7.1 tensorflow实现
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
##### 2.7.2 Softmax损失函数
```
def softmax_loss(z, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - z: Input data, of shape (N, C) where z[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dz: Gradient of the loss with respect to z
    """
    probs = np.exp(z - np.max(z, axis=1, keepdims=True))     # 1
    probs /= np.sum(probs, axis=1, keepdims=True)            # 2
    N = z.shape[0]                                            # 3
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N        # 4
    dz = probs.copy()
    dz[np.arange(N), y] -= 1
    dz /= N
    return loss, dz
```
##### 2.7.3 Softmax损失函数代码详解
- softmax_loss(z, y) 函数的输入数据是shape为(N, C)的矩阵z和shape为(N, )的一维array行向量y。由于损失函数的输入数据来自神经网络的输出层，所以这里的矩阵z中的N代表是数据集样本图片的个数，C代表的是数据集的标签个数，对应于CIFAR-10的训练集来说，z矩阵的shape应该为(50000, 10)，其中矩阵元素数值就是CIFAR-10的训练集数据经过整个神经网络层到达输出层，对每一张样本图片(每行)打分，给出对应各个标签(每列)的得分分数。一维array行向量y内的元素数值储存的是训练样本图片数据源的正确标签，数值范围是 0⩽yi<C=10，亦即 yi=0,1,…,9。

- 前2行代码定义了probs变量。
1. 首先，np.max(z, axis=1, keepdims=True) 是对输入矩阵x在横向方向挑出一个最大值，并要求保持横向的维度输出一个矩阵，即输出为一个shape为(N, 1)的矩阵，其每行的数值表示每张样本图片得分最高的标签对应得分；</br>
1. 然后，再 np.exp(z - ..) 的操作表示的是对输入矩阵z的每张样本图片的所有标签得分都被减去该样本图片的最高得分，换句话说，将每行中的数值进行平移，使得最大值为0；</br>
1. 再接下来对所有得分取exp函数，然后在每个样本图片中除以该样本图片中各标签的总和(np.sum)，最终得到一个与矩阵z同shape的(N, C)矩阵probs。</br>
上述得到矩阵probs中元素数值的过程对应的就是softmax函数：</br>

![softmax_1.png](https://i.imgur.com/p7VAnY6.png)

其中，我们已经取定了C的值：logC=−maxx<sub>i</sub>z<sub>ij</sub>，且z<sub>ij</sub>(z;W,B)对应于代码中的输出数据矩阵x的第i行、第j列的得分z[i, j]，其取值仅依赖于从输出层输入来的数据矩阵z和参数(W,B)，同理，S<sub>ij</sub> 表示矩阵probs的第i行、第j列的新得分。

举一个简单3个图像样本，4个标签的输入数据矩阵x的栗子来说明得分有着怎样的变化：

![softmax_2.png](https://i.imgur.com/4IbLyfB.png)

##### 2.7.4 定义损失函数
![softmax_3.png](https://i.imgur.com/gtbgoRM.png)

##### 2.7.5 定义损失函数例子
![softmax_4.png](https://i.imgur.com/VGmaECe.png)

##### 2.7.6 sotfmax得分矩阵的梯度
![softmax_5.png](https://i.imgur.com/wMjzjLL.png)

##### 2.7.7 sotfmax得分矩阵的梯度例子
![softmax_6.png](https://i.imgur.com/htRtyft.png)

### 2.8 激活函数总结
1. use ReLU. Be careful with your learning rates
2. try out Leaky ReLU/MaxOut/ELU
3. try out tanh but don't expect much
4. dot't use sigmoid

### 3.0 Q&A
### 3.1 为什么ReLu要好过于tanh和sigmoid function?

第一，采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。

第二，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失），从而无法完成深层网络的训练。

第三，Relu会使一部分神经元的输出为0，这样就造成了网络的**稀疏性**，并且减少了参数的相互依存关系，**缓解了过拟合**问题的发生。

第四，sigmoid和tanh的gradient在饱和区域非常平缓，接近于0，很容易造成vanishing gradient的问题，减缓收敛速度。vanishing gradient在网络层数多的时候尤其明显，是加深网络结构的主要障碍之一。相反，Relu的gradient大多数情况下是常数，有助于解决深层网络的收敛问题。

### 3.2 ReLU深度网络能逼近任意函数的原因

将激活空间分割/折叠成一簇不同的线性区域，像一个真正复杂的折纸。

事实证明，有足够的层，可以近似“平滑”任何函数到任意程度。 此外，如果在最后一层添加一个平滑的激活函数，会得到一个平滑的函数近似。

一般来说，我们不想要一个非常平滑的函数近似，它可以精确匹配每个数据点，并且过拟合数据集，而不是学习一个在测试集上可正常工作的可泛化表示。 通过学习分离器，我们得到更好的泛化性，因此ReLU网络在这种意义上更好地自正则化。

### 3.3 ReLUs用法
```
# tf.nn.relu() 放到隐藏层
# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
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
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# Hidden Layer with ReLU activation function
# 隐藏层用 ReLU 作为激活函数
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
output = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
```
参考：
1. [如何理解ReLU activation function?](https://www.zhihu.com/question/59031444/answer/177786603 "如何理解ReLU activation function?")
1. [为什么在生成对抗网络(GAN)中，隐藏层中使用leaky relu比relu要好？](https://www.zhihu.com/question/68514413/answer/268088852)
1. [ReLU深度网络能逼近任意函数的原因](https://zhuanlan.zhihu.com/p/23186434)
1. [激活函数(ReLU, Swish, Maxout)](https://www.cnblogs.com/makefile/p/activation-function.html)