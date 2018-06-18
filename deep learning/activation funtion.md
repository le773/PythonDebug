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

### 2.6 激活函数总结
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