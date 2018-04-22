## 第一周 卷积神经网络
### 1.6 卷积为什么有效？

![why_convolution_effective_1.png](https://i.imgur.com/S95bCXM.png)

### 1.7 单层卷积网络

![single_convolution_1.png](https://i.imgur.com/vqWTt2B.png)

### 1.11 为什么使用卷积？
和只用全连接层相比卷积层的两个主要优势在于**参数共享**和**稀疏连接**，可以减少过拟合。

卷积参数少的原因：
1. 参数共享(parameter sharing)

特征检测如垂直边缘检测如果适用于图片的某个区域，那么它也可能适用于图片的其它区域。

也就是说，如果用一个`3x3`的过滤器检测垂直边缘，那么图片的左上角区域以及旁边的各个区域都可以使用这个`3x3`的过滤器，每个特征检测器以及输出都可以在图片的不同区域中使用同样的参数，以便提取垂直边缘或其它特征。它不仅适用于边缘特征这样的低阶特征，同样适用于高阶特征。

2. 稀疏连接
![why_convolution_1.png](https://i.imgur.com/GZMnHib.png)

`In each layer, each output value depends only on a small number of inputs`


## 第二周 深度卷积网络：实例探究
### 2.2 经典网络
#### 2.2.1 LeNet-5

![LeNet-5.png](https://i.imgur.com/hC4rMIN.png)

`LeNet-5`卷积网络中，总共大约有6万个参数。随着网络层次加深，图像的宽度和高度都在减小，信道数量一直在增加。

#### 2.2.2 AlexNet

![AlexNet_1.png](https://i.imgur.com/kVlnwgi.png)

##### `AlexNet`和`LeNet-5`区别联系

1. `AlexNet`和`LeNet-5`有些类似，但`AlexNet`大约有6千万个参数。
2. 当用于训练图像和数据集时，`AlexNet`能够处理非常相似的基本构造模块，这些模块往往包含大量的隐藏单元或数据，这一点`AlexNet`表现出色
3. `AlexNet`使用了`ReLU`激活函数。

#### 2.2.3 VGG-16
![VGG-16_1.png](https://i.imgur.com/zGBI2mc.png)

`VGG-16`是一种只需要专注于构建卷积层的简单网络，总共包含1.38亿个参数。它**简化了神经网络结构**，每卷积一次**信道数量翻倍**，通过池化过程来压缩数据，宽度和高度都减小一倍。

主要的缺点：需要训练的特征数量非常巨大。

16指在这个网络包含16个卷积层和全连接层。

### 2.3 Residual Network 残差网络
残差网络，可以用来解决梯度消失和梯度爆炸

神经网络计算主要路径:

![Residual_block_2.png](https://i.imgur.com/v38ul1o.png)

a<sup>[l]</sup> 插入的时机是在线性激活之后，`ReLU`之前
##### 捷径`short cut`
![Residual Block](https://i.imgur.com/49dxUSJ.png)

在残差网络中，通过捷径直接把a<sup>[l]</sup>插入到第二个`ReLU`前：

![Residual_block_3.png](https://i.imgur.com/8RUxloT.png)

其中a<sup>[l]</sup>需要乘以权重w<sub>s</sub>使得它的的大小和z<sup>[l+2]</sup>匹配。

##### 跳远网络
a<sup>[l]</sup>跳过好几层网络，从而将信息传递到神经网络的更深处。

通过这种跳跃网络层的方式，获得更好的训练效果，这种结构构成残差块。

#### 2.3.2 残差网络
![Residual_block_4.png](https://i.imgur.com/zM6SEJf.png)

对普通网络训练越深，用优化算法越难训练，训练错误会越来越多。残差网络可以解决此问题。

![Residual_block_5.png](https://i.imgur.com/Wq4lgX5.png)

### 2.4 残差网络为什么有用？
一个网络深度越深，它在训练集上训练网络的效率会有所减弱，这是不希望加深网络的原因。但至少在训练`Residual Net`网络时，并不完全如此。

如果使用`L2`正则化或权重衰减，它会压缩w<sup>[l+2]</sup>的值，如果对b<sup>[l+2]</sup>也可达到同样的效果。

![Residual_block_6.png](https://i.imgur.com/Q82n6xl.png)

结果表明，残差块学习这个恒等式函数残差块并不难，跳远连接很容易得出a<sup>[l]</sup> = a<sup>[l+2]</sup>，这意味着即使给神经网络增加了这两层，它的效率并不逊色更简单的神经网络，因为学习恒等函数对它来说很简单，尽管增加了两层，也只是把a<sup>[l]</sup>赋值给a<sup>[l+2]</sup>，在大型神经网络中，不论残差块添加到神经网络中间还是末端位置，都不影响网络的表现。

对于一个神经网络中存在的一些恒等函数（`Identity Function`），残差网络在不影响这个神经网络的整体性能下，使得对这些恒等函数的学习更加容易，而且很多时候还能提高整体的学习效率。

![Residual_block_7.png](https://i.imgur.com/3nRJi1a.png)

这个网络有很多层3*3卷积，而且它们大多数都是相同的，这就是添加等维特征向量(w<sub>s</sub>)的原因，维度得以保留。