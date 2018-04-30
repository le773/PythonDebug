### 导论

通常统计学会有这样的两难问题：如果要得到一个准确的分布，首先需要较充足的样本量，然后如果样本太少的时候，又期望通过模型来生成样本，所以就有了EM算法。

1. 首先，通过E步骤，用极大似然估计来估计先验是小样本下的分布，
1. 然后用M计算在先验样本分布下的的后验样本，
1. 再用这些样本去走E步骤，不断迭代直到收敛。
1. 最终得到了一个精确的分布和一堆生成的样本。

### What is an intuitive explanation of the Expectation Maximization technique?

假设我们有一些数据来自两个不同的组，红色和蓝色：

![svm_muti_classification_1.png](https://i.imgur.com/6P5gWXM.png)

通过上图，可以观察到每个数据点属于红色或蓝色组，这样就可以很轻松找个每个组的参数。举例，红色组的均值大约是3，蓝色组的均值大约是7（如果我们想的话，我们可以找到确切的方法）。

一般来说，这就是所谓的最大似然估计。给定一些数据，我们计算一个参数（或参数）的值，以最好地解释数据。

现在想象，不知道数据点属于哪个分组，如下：

![uvcRO.png](https://i.imgur.com/pjEvDEC.png)

我们知道有两个分组的值，但不清楚每个值来自哪个组。我们还能找到估算出最适合我们所看到的数据的红蓝组均值的方法吗？

是的，EM算法提供了估算的方法，这个算法背后的基本思想是：

1. 用估值对每个参数初始化。
2. 计算每个参数产生数据点的概率(E)。
3. 根据参数产生的概率计算每个数据点的权重。
4. 将这些权重与数据结合起来，以计算对参数的更好估计(M)。
5. 重复步骤2到4，直到参数估计收敛(进程停止产生不同的估计)。

其中一些步骤需要进一步解释。最好是用一个例子和图片来完成。

例子：估计均值和标准偏差

假设我们有两组：红色和蓝色——这些值分布在上面的图像中。具体地说，每个组包含一个从正态分布中提取的值，其中包含以下参数：
```
import numpy as np
from scipy import stats

np.random.seed(110) # for reproducible random results

# set parameters
red_mean = 3
red_std = 0.8

blue_mean = 7
blue_std = 2

# draw 20 samples from normal distributions with red/blue parameters
red = np.random.normal(red_mean, red_std, size=20)
blue = np.random.normal(blue_mean, blue_std, size=20)

# 把两个分布的数据合并到一起
both_colours = np.sort(np.concatenate((red, blue))) 
# for later use...
```

当我们看到每个点的颜色（即它所属的组）时，很容易估计每个组的平均值和标准偏差。把红色和蓝色的值传递给NumPy中的内置函数，例如：

观察生成的红色、蓝色组正太分布参数：
```
>>> np.mean(red)
2.802
>>> np.std(red)
0.871
>>> np.mean(blue)
6.932
>>> np.std(blue)
2.195
```

但是如果我们不能看到这些点的颜色呢？也就是说，不是红色或蓝色，每个点对我们来说都是紫色的。

为了尝试恢复红蓝组的均值和标准偏差参数，我们可以使用EM。

步骤1，猜测每个组的平均值和标准偏差的参数值。我们不需要猜测很准确，选择任何数字均可：
```
# estimates for the mean
red_mean_guess = 1.1
blue_mean_guess = 9

# estimates for the standard deviation
red_std_guess = 2
blue_std_guess = 1.7
```

这些参数估计产生的曲线是这样的：

![GxLlr.png](https://i.imgur.com/0CkVi3g.png)

这些都是不好的估计：例如，这两种方法都远远偏离了任何一组点的均值。我们想要改进这些估计。

步骤2，计算在当前参数猜测下出现的每个数据点的概率：
```
likelihood_of_red = stats.norm(red_mean_guess, red_std_guess).pdf(both_colours)
likelihood_of_blue = stats.norm(blue_mean_guess, blue_std_guess).pdf(both_colours)
```

在这里，我们简单地把每个数据点放入概率密度函数中，我们现在猜测的是红色和蓝色的均值和标准差。例如，根据我们目前的猜测点1.761与蓝色(0.00003)相比更可能是红色(0.189)。

对于每个数据点，我们可以将这两个似然值转换为权重(步骤3)，以便它们之和为1：

```
likelihood_total = likelihood_of_red + likelihood_of_blue

red_weight = likelihood_of_red / likelihood_total
blue_weight = likelihood_of_blue / likelihood_total
```

根据我们目前的估计和新计算的权重，现在可以计算红蓝组的均值和标准差的新估计(第4步)。

我们用所有的数据点来计算平均值和标准偏差，但是有不同的权重：一次是红色的，一次是蓝色的。

直觉上关键一点是，在数据点上，颜色的权重越大，数据点就越会影响到对该颜色参数的下一个估计。这就产生了将参数"拉"到正确方向的效果。

```
def estimate_mean(data, weight):
    return np.sum(data * weight) / np.sum(weight)

def estimate_std(data, weight, mean):
    variance = np.sum(weight * (data - mean)**2) / np.sum(weight)
    return np.sqrt(variance)

# new estimates for standard deviation
blue_std_guess = estimate_std(both_colours, blue_weight, blue_mean_guess)
red_std_guess = estimate_std(both_colours, red_weight, red_mean_guess)

# new estimates for mean
red_mean_guess = estimate_mean(both_colours, red_weight)
blue_mean_guess = estimate_mean(both_colours, blue_weight)
```

我们对这些参数有了新的估计。为了再次改进它们，我们可以跳转到步骤2并重复这个过程。我们这样做，直到估算收敛，或者在执行了一些迭代之后(步骤5)。

对于我们的数据，这个过程的前五个迭代看起来是这样的(最近的迭代更接近真实分布):

![QbKmW.png](https://i.imgur.com/FvEsCpj.png)

我们看到，方法已经在一些值上收敛，曲线的形状(由标准偏差控制)也变得更加稳定。

如果继续迭代20次，将以下分布结束：

![RVFBC.png](https://i.imgur.com/h9QQPYn.png)

EM已经聚合到以下值，结果非常接近实际值(在这里我们可以看到颜色——没有隐藏的变量)：

```
          | EM guess | Actual |  Delta
----------+----------+--------+-------
Red mean  |    2.910 |  2.802 |  0.108
Red std   |    0.854 |  0.871 | -0.017
Blue mean |    6.838 |  6.932 | -0.094
Blue std  |    2.227 |  2.195 |  0.032
```

参考：
[What is an intuitive explanation of the Expectation Maximization technique?](https://stackoverflow.com/questions/11808074/what-is-an-intuitive-explanation-of-the-expectation-maximization-technique "What is an intuitive explanation of the Expectation Maximization technique?")

[怎么通俗易懂地解释EM算法并且举个例子?](https://www.zhihu.com/question/27976634/answer/153567695 "怎么通俗易懂地解释EM算法并且举个例子?")

[什么是Expectation Maximization?](https://www.zhihu.com/question/23413925/answer/24502810 "什么是Expectation Maximization?")
