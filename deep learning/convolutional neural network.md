### 1.0 概念
池化:把很多数据用最大值或者平均值代替。目的是降低数据量。

卷积:把数据通过一个卷积核变化成特征，便于后面的分离。计算方式与信号系统中的相同。

其连续的定义为：

![cnn_1.png](https://i.imgur.com/bL3R8MR.png)

其离散的定义为：

![cnn_2.png](https://i.imgur.com/bm9sH7u.png)

#### 卷积的物理意义
一个函数（如：单位响应）在另一个函数（如：输入信号）上的加权叠加。

### 2.0 统计不变性
即基本上不会随时间或空间改变的事物。

- 平移不变性

如果人们选择图像中的连续范围(照片中猫咪)作为池化区域，并且只是池化相同(重复)的隐藏单元产生的特征，那么，这些池化单元就具有平移不变性 (`translation invariant`)。

这就意味着即使图像经历了一个小的平移之后，依然会产生相同的 (池化的) 特征。

在很多任务中 (例如物体检测、声音识别)，我们都更希望得到具有平移不变性的特征，因为即使图像经过了平移，样例(图像)的标记仍然保持不变。

- 权重共享

当知道两个输入可能包含相同类型的信息时，通用它们的权重，并利用这些输入共同训练权重。

### 3.0 卷积网络
`CovNet`是一种空间上共享参数的神经网络。

通常在设计一个卷积网络的结构时，需要考虑卷积过程、池化过程的滤波器的大小，甚至是要不要使用`1×1`卷积核。

![cov_neural_net_1.png](https://i.imgur.com/SPnXU7y.png)

上图，在最右端定义一个分类器，所有空间信息被压缩成一个表示，仅映射到图片内容的参数被保留。

##### 3.1.1 CNN层次相关概念

![cov_neural_net_2.png](https://i.imgur.com/8G70ndB.png)

----------

##### 3.1.2 CNN学习实例图解

![cov_neural_net_3.png](https://i.imgur.com/iqyppiO.jpg)

CNN可能有几层网络，每个层可能捕获对象抽象层次中的不同级别。
- 第一层是抽象层次的最底级，CNN 一般把图片中的较小的部分识别成简单的形状，例如水平、竖直的直线，简单的色块。
- 下一层将会上升到更高的抽象层次，一般会识别更复杂的概念，例如形状（线的组合），
- 以此类推直至最终识别整个物体，例如狗。

再次强调，CNN 是**自主学习**。我们不需要告诉 CNN 去寻找任何直线、曲线、鼻子、毛发等等。CNN 从训练集中学习并发现金毛巡回犬值得寻找的特征。

### 4.0 滤波器 Filters
#### 4.1 分解一张图片

CNN 的第一步是把图片分成小块。我们通过选取一个给定宽度和高度的滤波器来实现这一步。

滤波器会照在图片的小块`patch`（图像区块）上。这些`patch`的大小与滤波器一样大。

![cov_neural_net_4.png](https://i.imgur.com/8SAKOp5.png)

滤波器滑动的间隔被称作 `stride`（步长）。这是你可以调节的一个超参数。增大 `stride` 值后，会减少每层总 `patch` 数量，因此也减小了模型大小。通常这也会降低图像精度。

#### 4.2 滤波器深度 Filter Depth
通常都会有多余一个滤波器，不同滤波器提取一个 `patch` 的不同特性。例如，一个滤波器寻找特定颜色，另一个寻找特定物体的特定形状。**卷积层滤波器的数量**被称为滤波器深度。

##### 4.2.1 3x3滤波器

![neural_net_7.png](https://i.imgur.com/DVATKCm.gif)

##### 4.2.2 卷积的计算
![conv_1.png](https://i.imgur.com/HTXXAbr.png)

输入x(N, C, H, W)，卷积核f(F, C, HH, WW)，1a.对每一个卷积核，将卷积核f[0,1,:,:]与x[0,0,:,:]做矩阵的点积，1b.然后计算所有通道上点积的和加偏置项，即为out[0,0,0]。然后在x[0,:,:,:]上移动步长stride计算重复1a，1b得到out[0,0,1]。

##### 4.2.3 卷积的计算代码实现
```python
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pass
  pad, stride = conv_param['pad'], conv_param['stride']
  # 对第三、四维做0值填充
  x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant') # pad alongside four dimensions
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  output_height = 1 + (H + 2 * pad - HH) // stride
  output_width = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, output_height, output_width))

  for i in range(output_height):
      for j in range(output_width):
          # 截取需要卷积的区域
          x_padded_mask = x_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(F):#遍历权重
              out[:, k, i, j] = np.sum(x_padded_mask * w[k, :, :, :], axis=(1,2,3)) # 对所有通道上的点积求和
  out = out + (b)[None, :, None, None]
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache
```
##### 4.2.4 卷积的反向传播

![covnbp.jpg](https://i.imgur.com/fNLCOEn.jpg)

##### 4.2.5 卷积的反向传播代码实现
```python
def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  dx = np.zeros_like(x)
  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  db = np.sum(dout, axis = (0,2,3))

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  for i in range(int(H_out)):
      for j in range(int(W_out)):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(F): #compute dw
              dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
          for n in range(N): #compute dx_pad
              dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] *
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)
  dx = dx_pad[:,:,pad:-pad,pad:-pad]
  pass
############################################################################
  return dx, dw, db
```
##### 4.2.6 每个 patch 连接多少神经元？
这取决于滤波器的深度，如果深度是 `k`，我们把每个 `patch` 与下一层的 `k` 个神经元相连。这样下一层的高度就是 `k`，如下图所示。实际操作中，`k`是一个我们可以调节的超参数，大多数的 `CNNs` 倾向于选择相同的起始值。

![cov_neural_net_5.png](https://i.imgur.com/0TXnm1f.png)

一个 `patch` 连接有**多个神经元**可以保证我们的 `CNNs` 学会**提取任何它觉得重要的特征**。
记住，`CNN` 并没有被规定寻找特定特征。与之相反，它自我学习什么特征值得注意。

##### 4.2.7 为什么我们把一个 patch 与下一层的多个神经元相连呢？一个神经元不够好吗？
多个神经元的作用在于，一个 patch 可以有多个有意义的，可供提取的特点。
例如，一个 patch 可能包括白牙，金色的须，红舌头的一部分。在这种情况下，我们需要一个深度至少为3的滤波器，一个识别牙，一个识别须，一个识别舌头。

#### 4.3 卷积续
全链接层是一个标准的，非卷积层。它的输入与所有的输出神经相连，也被称为 dense 层

#### 4.4 卷积的导数
![cnn的导数.png](https://i.imgur.com/DWhvlaW.png)

和深度神经网络没有什么差别，除了求导对象变为卷积核。

### 5.0 参数
#### 5.1 参数共享
![cov_neural_net_share_param_1.png](https://i.imgur.com/Jw7lNgU.png)

当我们试图识别一个猫的图片的时候，我们并不在意猫出现在哪个位置。无论是左上角，右下角，它在你眼里都是一只猫。我们希望 CNNs 能够无差别的识别，这如何做到呢？

如我们之前所见，一个给定的 `patch` 的分类，是由 `patch` 对应的**权重和偏置**项决定的。

如果我们想让左上角的猫与右下角的猫以同样的方式被识别，他们的权重和偏置项需要一样，这样他们才能以同一种方法识别。

这正是我们在 CNNs 中做的。**一个给定输出层学到的权重和偏置项会共享在输入层所有的 patch 里**。注意，当我们增大滤波器的深度的时候，我们需要学习的权重和偏置项的数量也会增加，因为权重并没有共享在所有输出的 channel 里。

共享参数还有一个额外的好处。如果我们不在所有的 patch 里用相同的权重，我们必须对每一个 patch 和它对应的隐藏层神经元学习新的参数。这不利于规模化，特别对于高清图片。因此，共享权重不仅帮我们平移不变，还给我们一个更小，可以规模化的模型。

#### 5.2 Padding
![cov_neural_net_padding_1.png](https://i.imgur.com/TjrYN5u.png)

一个 `5x5` 的网格附带一个 `3x3` 的滤波器

假设现在有一个 `5x5` 网格 (如上图所示) 和一个尺寸为 `3x3 stride`值为 `1` 的滤波器(`filter`)。 下一层的 `width` 和 `height` 是多少呢？ 如图中所示，在水平和竖直方向都可以在`3`个不同的位置放置 `patch`， 下一层的维度即为 `3x3` 。下一层宽和高的尺寸就会按此规则缩放。

在理想状态下，我们可以在层间保持相同的宽度和高度，以便继续添加图层，保持网络的一致性，而不用担心维度的缩小。如何实现这一构想？其中一种简单的办法是，在 `5x5` 原始图片的外层包裹一圈 `0` ，如下图所示。

![cov_neural_net_padding_2.png](https://i.imgur.com/VI8j7ts.png)

加了 0 padding的相同网格。

这将会把原始图片扩展到 `7x7`。 现在我们知道如何让下一层图片的尺寸维持在 `5x5`，保持维度的一致性。

#### 5.3 维度
综合目前所学的知识，我们应该如何计算 `CNN` 中每一层神经元的数量呢？

- 输入层（`input layer`）维度值为`W`， 滤波器（`filter`）的维度值为 `F` (`height * width * depth`)， `stride` 的数值为 `S`， `padding` 的数值为 `P`， 下一层的维度值可用如下公式表示: `(W−F+2P)/S+1`(向下取整)。

- 新的深度就是滤波器的数量。

- 滤波器中通道的数量必须与输入中通道的数量一致

我们可以通过每一层神经元的维度信息，得知模型的规模，并了解到我们设定的 `filter size` 和 `stride` 如何影响整个神经网络的尺寸。

##### 5.3.1 SAME Padding
```
out_height = ceil(float(in_height) / float(strides1))
out_width = ceil(float(in_width) / float(strides[2]))
```
##### 5.3.2 VALID Padding
```
out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
```

##### 5.3.3 代码实例
```
import tensorflow as tf
input = tf.placeholder(tf.float32,(None, 32, 32, 3))
# (height, width, input_depth, output_dep
filter_weights = tf.Variable(tf.truncated_normal((8,8,3,20)))
filter_bias = tf.Variable(tf.zeros(20))
# (batch, height, width, depth)
strides = [1,2,2,1]
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
print(conv)
```
#### 5.4 没有参数共享
没有参数共享，每个输出层的神经元必须连接到滤波器的每个神经元。此外，每个输出层的神经元必须连接到一个偏置神经元。
```
输入数据，维度为 32x32x3 (HxWxD)
20个滤波器，维度为 8x8x3 (HxWxD)
stride（步长）高和宽的都为 2 (S)
padding 大小为1 (P)
输出层:14x14x20 (HxWxD)

答：
(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560
8 * 8 * 3 是权值数量，加上1作为 bias。因为每一个权值都与输出的每一部分相连。
```
#### 5.5 参数共享
有了参数共享，每个输出通道的神经元与相同通道的其它神经元共享权值。参数的数量与滤波器神经元的数量相同，加上偏置，再乘以输出层的通道数。
```
答：(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860
```

### 6.0 CNNs可视化
### 7.0 探索设计空间
池化层的输出深度与输入的深度相同。另外池化操作是分别应用到每一个深度切片层。

#### 7.1 最大池化
**优点**
1. 不会增加参数数量，所以不必担心导致容易过拟合
2. 通常会提高模型的准确性，
由于在非常小的步幅下进行卷积，模型必然需要更多的计算量，而且有更多的超参数需要调整，例如池区尺寸和池化步幅，它们不必完全相同。

一种典型的卷积神经网络结构为卷积层和最大池化层，相互交替，然后在最末端连接几层全连接层。

#### 7.2 最大池化代码实现
```python
def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  out_height = H // pool_height
  out_width = W // pool_width
  out = np.zeros((N, C, out_height, out_width))
  for i in range(out_height):
      for j in range(out_width):
          mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          out[:, :, i, j] = np.max(mask, axis=(2, 3))
  ############################################################################
  cache = (x, pool_param)
  return out, cache
```
#### 7.3 最大池化反向求导代码实现
```python
def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  dx = np.zeros_like(x)
  out_height = H // pool_height
  out_width = W // pool_width
  for i in range(out_height):
      for j in range(out_width):
          # x, dx has the same dimension, so does x_mask and dx_mask
          x_mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          dx_mask = dx[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          # flags: only the max value is True, others are False
          flags = np.max(x_mask, axis=(2, 3), keepdims=True) == x_mask
############################################################################
  return dx
```
最大池化的代码实现和反向求导实现的原理和dropout的实现与反向传播思想异曲同工。

#### 7.4 平均池化
使用特定位置周围的像素的平均值，它有点像提供了下层特征图的一个低分辨率的视图。

##### 例子
```
input = tf.placeholder(tf.float32,(None,4,4,5))
filter_shape = [1,2,2,1]
strides = [1,2,2,1]
padding = 'VALID'
pool = tf.nn.avg_pool(
    input,
    filter_shape,
    strides,
    padding)
print(pool)
```
#### 7.5 池化函数的导数
![pool的导数.png](https://i.imgur.com/sgvGxzK.png)

通过每一层的函数对函数的求导，可求得参数的梯度。
有了计算梯度的方法，在通过基于梯度的最优化，就能寻得最优值，完成训练过程。

### 8.0 TensofFlow 最大池化
![max_pooling_1.png](https://i.imgur.com/eKB41R5.png)

```
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1], # 滤波器大小
    strides=[1, 2, 2, 1], # 步长
    padding='SAME')
```
`ksize` 和 `strides` 参数也被构建为四个元素的列表，每个元素对应 `input tensor` 的一个维度 `([batch, height, width, channels])`，对 `ksize` 和 `strides` 来说，`batch` 和 `channel` 通常都设置成 `1`。


近期，池化层并不是很受青睐。部分原因是：
- 现在的数据集又大又复杂，我们更关心欠拟合问题。
- `Dropout` 是一个更好的正则化方法。
- 池化导致信息损失。想想最大池化的例子，n 个数字中我们只保留最大的，把余下的 n-1 完全舍弃了。

#### 8.1 例子
![pooling_mechanics_quiz_1.png](https://i.imgur.com/3EppHdJ.jpg)

这里，最大池化滤波器的大小是 `2x2`。当最大池化层在输入层滑动时，输出是这个 `2x2` 方块的最大值。

### 9.0 1x1 卷积
![cov_neural_net_6.png](https://i.imgur.com/kVTxpwt.png)

##### 为什么会有人想用`1x1`卷积？

因为它们关注的不是一块图像，而仅仅是一个像素。传统的的卷积，基本上是运行在一小块图像上的小分类器，但仅仅是个线性分类器，但如果在中间加一个`1x1`卷积，就用运行在一块图像上的神经网络代替了线性分类器，在卷积操作中散布一些`1x1`卷积，是一种使**模型变得更深的低耗高效**的方法，并且会有更多的参数，但未完全改变神经网络结构，它们非常简单，因为如果看数学公式，它们根本不是卷积，只是矩阵相乘并且仅有较少的参数。

未使用`1x1`滤波器的卷积计算量为120w

![conv_cal_1.png](https://i.imgur.com/lVg13Ja.png)

使用`1x1`滤波器的卷积计算量为12.4w

![conv_cal_1x1_1.png](https://i.imgur.com/d5QZjt7.png)

**总结**：只要合理构建瓶颈层，既可以显著缩小表示层规模，又不会降低网络性能，从而节省了计算量。

池化压缩数据的高度和宽度，`1x1`卷积核能压缩数据的信道数。

### 10.0 Inception 模块
#### 10.1 Inception 模块
`Inception`网络或`Inception`层的作用是:代替人工来确定**卷积中的过滤类型**，或者确定是否需要创建**卷积层**或**池化层**。

**基本思想**：`Inception`网络不需要人为决定使用哪个滤波器，或者是否需要池化，而是由网络自行确定这些参数。

![nn_inception_1.png](https://i.imgur.com/LY3iQR3.png)

`Inception`不局限于单个卷积运算，而是将多个模块组合，如平均池化后接`1x1`卷积等，最后把这些运算输出连成一串。看起来很复杂，根据选择参数的方式，模型中的参数总数可能非常少，但模型的性能比使用简单卷积时要好。

#### 10.2 Inception 网络
![nn_inception_2.png](https://i.imgur.com/l5srf2w.png)

### 11.1 卷积神经网络优缺点
##### 优点
1. 共享卷积核(共享参数)，对高维数据的处理没有压力
2. 无需选择特征属性，只要训练好权重，即可得到特征值
3. 深层次的网络抽取图像信息比较丰富，表达效果好
##### 缺点
1. 需要调参，需要大量样本，训练迭代次数比较多，最好使用GPU训练
2. 物理含义不明确，从每层输出中很难看出含义来

参考：

[如何理解卷积神经网络（CNN）中的卷积和池化？](https://www.zhihu.com/question/49376084)

[卷积为什么叫「卷」积？](https://www.zhihu.com/question/54677157/answer/141245297 "卷积为什么叫「卷」积？")
