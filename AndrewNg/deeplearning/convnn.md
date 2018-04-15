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
#### LeNet-5 
每个过滤器都采用和输入模块一样的信道数量。

### 2.3 Residual Network 残差网络
