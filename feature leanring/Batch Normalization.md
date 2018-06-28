### 导论
统计机器学习中有一个经典的假设:Source Domain 和 Target Domain的数据分布是一致的。也就是说，训练数据和测试数据是满足相同分布的。这是通过训练数据获得的模型能够在测试集上获得好的效果的一个基本保障。

Convariate Shift是指训练集的样本数据和目标样本集分布不一致时，训练得到的模型无法很好的Generalization。它是分布不一致假设之下的一个分支问题，也就是指Sorce Domain和Target Domain的条件概率一致的，但是其边缘概率不同。的确，对于神经网络的各层输出，在经过了层内操作后，各层输出分布就会与对应的输入信号分布不同，而且差异会随着网络深度增大而加大了，但每一层所指向的Label仍然是不变的。

解决办法：一般是根据训练样本和目标样本的比例对训练样本做一个矫正。所以，通过引入Bactch Normalization来标准化某些层或者所有层的输入，从而固定每层输入信息的均值和方差。

### 1.0 Batch Normalization
1. 把具有不同尺度的特征映射到同一个坐标系，具有相同的尺度(相似特征分布)，使激活函数分布在线性区间，结果就是加大了梯度，让模型更大胆的进行梯度下降。
2. 一定程度上消除了噪声、质量不佳等各种原因对模型权值更新的影响。
3. 破坏原来的数据分布，一定程度上缓解了过拟合。
4. 更容易跳出局部最小值。

含有batch-norm的神经网络计算步骤：

![mini-batch-norm_1.png](https://i.imgur.com/wFGzFb4.png)

对于含有m个节点的某一层神经网络，对z进行操作的步骤为:

![mini-batch-norm_2.png](https://i.imgur.com/sfNipn1.png)

第一，这里的常规归一化实际上就是改变了一个mini-batch中样本的分本，由原来的某个分布转化成均值为0方差为1的标准分布；</br>
第二，仅转化了分布还不行，因为转化过后可能改变了输入的取值范围，因此需要赋予一定的放缩和平移能力，即将归一化后的输入通过一个仿射变换的子网络。其中的`γ`、`β`并不是超参数，而是两个需要学习的参数，神经网络自己去学着使用和修改这两个扩展参数。这样神经网络就能自己慢慢琢磨出前面的标准化操作到底有没有起到优化的作用。如果没有起到作用，就使用`γ`、`β`来抵消一些之前进行过的标准化的操作。</br>
第三，这里的所有操作都是可微分的，也就使得了梯度后向传播算法在这里变得可行；

### 1.1 batch-norm代码实现
```python
def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.
  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:
  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var
  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.
  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5) # 数值稳定参数
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':# 训练模式
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0) # 每个特征的均值
    sample_var = np.var(x, axis = 0) # 每个特征的方差
    x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps)) # +eps防止分母为0，保持数值稳定

    out = gamma * x_hat + beta

    cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean # 动量更新法更新running_mean参数
    # 指数平滑法：最终测试用的running_mean, running_var参数不再是一个bn层决定的，而是所有
    # BN层一起决定
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #pass
    #############################################################################
  elif mode == 'test':# 测试模式
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    out = gamma * (x - running_mean) / (np.sqrt(running_var + eps)) + beta
    #pass
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache
```

### 1.2 Batch Normalization back propagation
![bpbn2.png](https://i.imgur.com/qkC4VT4.png)

### 1.3 反向传播代码实现
```python
def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x, mean, var, x_hat, eps, gamma, beta = cache
  N = x.shape[0]
  dgamma = np.sum(dout * x_hat, axis = 0) # 见印象笔记BN层反向传播第5行公式
  dbeta = np.sum(dout * 1.0, axis = 0) # 第6行公式
  dx_hat = dout * gamma # 第1行
  dx_hat_numerator = dx_hat / np.sqrt(var + eps) # 第3行第1项
  dx_hat_denominator = np.sum(dx_hat * (x - mean), axis = 0) # 第2行前半部分
  dx_1 = dx_hat_numerator # 第4行第1项
  dvar = dx_hat_denominator * (-0.5) * ((var + eps)**(-1.5)) #第2行
  dmean = -1.0 * np.sum(dx_hat_numerator, axis = 0) + dvar * np.mean((-2.0) * (x - mean) / N, axis = 0) # 第3行
  dx_var = dvar * 2.0 * (x - mean) / N # 第4行第2部分 
  dx_mean = dmean * 1.0 / N # 第4行第3部分 
  dx = dx_1 + dx_var + dx_mean # 第4行
  #############################################################################
  # gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
  # N = x.shape[0]

  # dx_1 = gamma * dout
  # dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
  # dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
  # dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
  # dx_4_b = dx_3_b * 1
  # dx_5_b = np.ones_like(x) / N * dx_4_b
  # dx_6_b = 2 * (x - u_b) * dx_5_b
  # dx_7_a = dx_6_b * 1 + dx_2_a * 1
  # dx_7_b = dx_6_b * 1 + dx_2_a * 1
  # dx_8_b = -1 * np.sum(dx_7_b, axis=0)
  # dx_9_b = np.ones_like(x) / N * dx_8_b
  # dx_10 = dx_9_b + dx_7_a

  # dgamma = np.sum(x_hat * dout, axis=0)
  # dbeta = np.sum(dout, axis=0)
  # dx = dx_10
  #pass
  #############################################################################

  return dx, dgamma, dbeta
```
### 2.0 Batch Norm为什么会奏效？
通过归一化所有的输入特征值x，以获得类似范围的值，可以加快学习。`Batch-norm`是类似的道理

当前的获得的经验无法适应新样本、新环境时，便会发生“`Covariate Shift`”现象。 对于一个神经网络，前面权重值的不断变化就会带来后面权重值的不断变化，批标准化减缓了隐藏层权重分布变化的程度。采用批标准化之后，尽管每一层的z还是在不断变化，但是它们的**均值和方差将基本保持不变，限制了在前层的参数更新会影响数值分布的程度，使得后面的数据及数据分布更加稳定，减少了前面层与后面层的耦合**，使得每一层不过多依赖前面的网络层，最终加快整个神经网络的训练。

### 2.1 When to use BN?
1. 在神经网络训练时遇到收敛速度很慢，或梯度爆炸等无法训练的状况时可以尝试BN来解决。
2. 另外，在一般使用情况下也可以加入BN来加快训练速度，提高模型精度。

参考：
1. [Batch Normalization原理及其TensorFlow实现](https://www.cnblogs.com/bonelee/p/8528722.html)
2. [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762)