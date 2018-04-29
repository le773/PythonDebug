### 01 Standard Large-Margin Problem
![standard_large-Margin_1.png](https://i.imgur.com/W4XaVnO.png)

因为点`x'`和`x''`在超平面![超平面公式](http://img.blog.csdn.net/20131107201104906)上，则有：
1. w<sup>T</sup> x' + b = 0       (1)
2. w<sup>T</sup> x'' + b = 0      (2)

`w`是平面 w<sup>T</sup>的垂直向量，由 (2) - (1) 得:
<center>w<sup>T</sup>(x'' - x') = 0</center>
那么，向量`(x'' - x')`于平面w<sup>T</sup>上；接下来计算向量`(x - x')`在向量`w`上的投影长度:

![standard_large-Margin_2.png](https://i.imgur.com/8cnTPld.png)

假设，超平面能将所有训练样本分类，那么：

![standard_large-Margin_3.png](https://i.imgur.com/q1UL4cn.png)

将上述公式，简写为一个公式：

![standard_large-Margin_4.png](https://i.imgur.com/z2yGJkf.png)

所有训练样本到超平面的距离可以表示为：

![standard_large-Margin_5.png](https://i.imgur.com/5lxMsrD.png)

考虑支持向量到平面的距离，即是支持向量`w<sup>T</sup> x + b = 1`到超平面的距离，推导如下：

![standard_large-Margin_6.png](https://i.imgur.com/usQlyUf.png)

求max(1/||w||)，可以等价于标准问题：min (0.5 * w * w<sup>T</sup>)

![standard_large-Margin_8.png](https://i.imgur.com/MeDSMnh.png)

下图的含义是，在训练样本中，样本到超平面最小的距离是1：

![standard_large-Margin_9.png](https://i.imgur.com/B1pR82z.png)

`support vector`：是寻找最佳超平面的样本，其它的样本对寻找最佳超平面是没有作用的。

![standard_large-Margin_10.png](https://i.imgur.com/KOwcg6v.png)

在下图中，虚线经过的点就是`support vector`

![standard_large-Margin_11.jpg](https://i.imgur.com/lMlaZ1g.jpg)


----------

### 02  Solving General SVM

#### 02.01 转化为凸二次规划

最大化`1/(w.T)`转化为约束最优化问题：

![凸二次规划](http://ww4.sinaimg.cn/large/6cbb8645gw1ewo7sn30ngj208m02ct8l.jpg)

平方项保证了表达式的单调性，它会放大结果但不会改变表达式的顺序。

----------

### 03 Why Large-Margin Hyperplane
`regularization`:测量误差最小，且满足ww<sup>T</sup> < C

`SVM`：最小化ww<sup>T</sup>，且满足训练样本无错分。

`SVM`和`regularization`目标和限制条件分别对调，其实，考虑的内容是类似的，效果也是相近的。`SVM`也可以说是一种`weight-decay` `regularization`且测量误差为0

![standard_large-Margin_12.png](https://i.imgur.com/a2Kcpcg.png)


### 04 Large-Margin Restricts Dichotomies
![standard_large-Margin_13.png](https://i.imgur.com/UpPnV7h.png)

不看虑软间隔，满足最小间隔距离的的分割面可能不存在。

### 05 Lagrange Function
首先定义原始目标函数`f(x)`，拉格朗日乘子法的基本思想是把约束条件转化为新的目标函数`L(x,α)`的一部分，从而使有约束优化问题变成我们习惯的无约束优化问题。

![Larange_f_1.png](https://i.imgur.com/FqAVMDI.png)

`αn`是第`n`个训练样本的拉格朗日乘子。

#### 05.01 最小问题包含最大问题

![Larange_f_2.png](https://i.imgur.com/uc7mcv4.png)

固定`b w`，增大或减小`α`，使拉格朗日函数最大。

如果`b w`不满足约束条件，则1 - y<sub>n</sub>(w<sup>T</sup> * z<sub>n</sub> + b ) > 0，如果要使使拉格朗日函数最大，则需要α更大，显然此时无法求得最小的SVM；

反之`b w`满足约束条件，那么当`α=0`时，`svm`得到最小值。

----------

### 06 Lagrange Dual Problem

![Larange_f_3.png](https://i.imgur.com/7zVveOh.png)

不等式右侧，固定`α'`改变`b w`取最小值，此时求得最小值时有`b1 w1`

那么对于不等式左侧，相当于固定`b1 w1`，更改`α`求得最大值，显然上述不等式成立。

![Larange_f_4.png](https://i.imgur.com/ji6BB8L.png)

上述不等式，右侧是svm问题的下界.

右侧：固定`α'`调整`b w`，求最小拉格朗日函数中的最大值

左侧：固定`b w`调整`α`，求最大拉格朗日函数中的最小值

对于不等式成立，右侧取得目标值时的`b w`，左侧取此参数下的取它的最大值

转化为对偶问题的优势：无约束条件

证明：

![Larange_f_5.png](https://www.zhihu.com/equation?tex=%5Ctheta_D%28%5Cboldsymbol%7B%5Calpha%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%29%3D%5Cmin_%7B%5Cboldsymbol%7Bx%7D%7DL%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Calpha%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%29+%5Cleq+L%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Calpha%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%29+%5Cleq+%5Cmax_%7B%5Cboldsymbol%7B%5Calpha%7D%2C%7E%5Cboldsymbol%7B%5Cbeta%7D%3B%7E%5Cbeta_j%5Cgeq0%7DL%28%5Cboldsymbol%7Bx%7D%2C%5Cboldsymbol%7B%5Calpha%7D%2C%5Cboldsymbol%7B%5Cbeta%7D%29+%3D+%5Ctheta_P%28%5Cboldsymbol%7Bx%7D%29)

已知≥是一种弱对偶关系，在二次规划`QP`问题中，如果满足以下三个条件：

![Larange_f_6.png](https://i.imgur.com/PfIpdvM.png)

那么，上述不等式关系就变成强对偶关系，`≥`变成`=`，即一定存在满足条件的解`(b,w,α)`，使等式左边和右边都成立，`SVM`的解就转化为右边的形式。

### 07 Solving Larange Dual:Simplifications
![Larange_f_7.png](https://i.imgur.com/kVUmrCt.png)

根据梯度下降算法思想：最小值位置满足梯度为零

对`b`求导，令`L`对参数`b`的梯度为零：

![Larange_f_8.png](https://i.imgur.com/sMXMzVQ.png)

得到：

![Larange_f_15.png](https://i.imgur.com/6sSAZVK.png)

![Larange_f_9.png](https://i.imgur.com/z7cOjT0.png)

去除`b`后的等式

![Larange_f_10.png](https://i.imgur.com/45KPruR.png)

对`w`求导，令`L`对参数`w`的梯度为零：

![Larange_f_11.png](https://i.imgur.com/LCj5H1s.png)

得到：

![Larange_f_14.png](https://i.imgur.com/NsrBR8O.png)

然后，将`w`带入上式：

![Larange_f_12.png](https://i.imgur.com/4nQfnOv.png)

拉格朗日对偶问题的简化版

![Larange_f_13.png](https://i.imgur.com/1rXuBc5.png)


### 08 Karush-Kuhn-Tucker conditions

![Larange_f_16.png](https://i.imgur.com/Ar3TCI3.png)

### 09 Dual Formulation of Support Vector Machine
将`max`问题转化为`min`问题:

![Larange_f_19.png](https://i.imgur.com/pdk10do.png)

根据对`w`，`b`求导，得到`w`，`b`的值，代入上述公式

![Larange_f_21.png](https://i.imgur.com/xQD5Qyb.png)

根据多项式乘法的基本规律（所有项和的积等于所有项积的和）

![Larange_f_22.png](https://i.imgur.com/9ty4ytV.png)

代入公式可得：

![Larange_f_20.png](https://i.imgur.com/TE6FCdw.png)

### 10 Dual SVM with QP Solver
![Larange_f_18.png](https://i.imgur.com/a7UrTjC.png)

求`α`的值

![Larange_f_23.png](https://i.imgur.com/pDa7J3d.png)

根据`α`求`w`，`b`

![Larange_f_24.png](https://i.imgur.com/rDVHw24.png)

### 11 核函数
现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本的超平面；对这样的问题，可将样本映射到更高维的特征空间，使得样本在这个特征空间内线性可分。

令`φ(x)`表示将`x`映射后的特征向量，也是在特征空间中划分超平面所对应的模型为：

![svm_kernel_2.png](https://i.imgur.com/h1PSsYr.png)

转化为最优化问题

![svm_kernel_3.png](https://i.imgur.com/HUV8u3c.png)

其对偶问题是

![svm_kernel_4.png](https://i.imgur.com/Lndal2a.png)

![svm_kernel_5.png](https://i.imgur.com/cYbOABq.png)

常用核函数

![svm_kernel_1.png](https://i.imgur.com/eMLCvZP.png)

#### 11.01 Linear Kernel
`Linear Kernel：K(x, x') = xTx'`

- 优点
`safe`（一般不太会`overfitting`，所以线性的永远是我们的首选方案）；
`fast`，可以直接使用`General SVM`的`QP`方法来求解，比较迅速；
`explainable`，可解释性较好，我们可以直接得到`w, b`，它们直接对应每个`feature`的权重。

- 缺点
restrict：如果是线性不可分的资料就不太适用了！
 

#### 11.02 Polynomial Kernel
`Polynomial Kernel: K(x, x') = (ζ + γxTx')Q`

- 优点
我们可以通过控制`Q`的大小任意改变模型的复杂度，一定程度上解决线性不可分的问题；

- 缺点
含有三个参数，太多啦！

#### 11.03 径向基核
使用径向基核前，应对`feature`进行归一化

![径向基核](http://img.my.csdn.net/uploads/201304/03/1364958259_8460.jpg)

#### 11.03.01 参数分析
- `γ`参数实际上对 `SVM` 的“线性”核函数没有影响。核函数的重要参数是`C`, 
- `C`参数
1. `C`越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱(过拟合)。 
2. `C`值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强(容易欠拟合);

- `degree`:多项式`poly`函数的维度;
- `kernel` ：核函数，默认是`rbf`，可以是`linear`, `poly`, `rbf`, `sigmoid`, `precomputed`

#### 11.03.02 RBF公式里面的sigma和gamma的关系

![rbf中的gamma](http://img.blog.csdn.net/20150606105930104?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

`gamma`是你选择径向基函数作为`kernel`后，该函数自带的一个参数。隐含地决定了数据**映射到新的特征空间后的分布**。

如果`gamma`设的太大，`σ`会很小，`σ`很小的高斯分布长得又高又瘦，会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让`σ`无穷小，则理论上，高斯核的`SVM`可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过训练；

而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。

#### 11.03.03 rbf的优势
建议首选`RBF`核函数进行高维投影，因为：

1. 能够实现非线性映射；（ 线性核函数可以证明是他的一个特例；`SIGMOID`核函数在某些参数上近似RBF的功能。）
2. 参数的数量影响模型的复杂程度，多项式核函数参数较多。
3. `the RBF kernel has less numerical difficulties.`

#### 11.03.04 总结
所以在实际应用中，一般是先使用线性的`kernel`，如果效果不好再使用`gaussian kernel`（小的`γ`）和多项式`kernel`(小的`Q`)。

[svm详解](https://www.cnblogs.com/little-YTMM/p/5547642.html "svm详解")


### 12 软间隔
#### 12.01 软间隔
现实任务中往往很难确定合适的核函数使得训练样本在特征空间中线性可分；即使恰好找到某个核函数使用训练集在特征空间中线性可分，也很难断定这个线性可分结果不是由于过拟合所造成。

缓解该问题的一个办法是允许支持向量机在一些样本上出错，为此，要引入软间隔的概念。

![svm_soft_margin_1.png](https://i.imgur.com/KFMTKhf.png)

在最大化间隔的同时，不满足约束的样本应尽可能少，于是，优化目标可以写为：

![svm_soft_margin_2.png](https://i.imgur.com/KCRyysx.png)

其中`C>0`是一个常数，损失函数的定义：

![svm_soft_margin_3.png](https://i.imgur.com/ruj8EvN.png)

如果采用`hinge`损失，优化目标可以写为：

![svm_soft_margin_4.png](https://i.imgur.com/ItMeXlK.png)

引入松弛变量，优化目标可以写为：

![svm_soft_margin_5.png](https://i.imgur.com/YUKRiTI.png)

![svm_soft_margin_6.png](https://i.imgur.com/IblvhPi.png)

引入松弛变量后的拉格朗日函数：

![svm_soft_margin_7.png](https://i.imgur.com/CHdUTOy.png)

令`L`对`w` `b` `ξ`求偏导数

![svm_soft_margin_8.png](https://i.imgur.com/XUQLemf.png)

对偶问题

![svm_soft_margin_9.png](https://i.imgur.com/525yyVN.png)

#### 12.02 常用替代损失函数
![svm_replace_f_1.png](https://i.imgur.com/gdV8914.png)

![svm_replace_f_2.png](https://i.imgur.com/IpWWx3J.png)

### 13 补充

支持向量与间隔的定义
![standard_large-Margin_7.png](https://i.imgur.com/wyTnBn8.png)

#### 13.01 SVM决策边界
![svm_decision_boundary_1.png](https://i.imgur.com/C8kiPw8.png)

上图，左侧支持向量在超平面切线上投影的距离小于右侧，那么`θl > θr`，所以超平面选择右侧。

#### 13.02 SVM多分类
![svm_muti_classification_1.png](https://i.imgur.com/im9dRcz.png)

和逻辑回归类似，训练`K`个`SVM`，针对每一个分类器训练得到一个`θ`，然后选择对此样本最好的那个参数。

#### 13.02 SVM LogisticRegression
![svm_vs_logisticregression_1.png](https://i.imgur.com/kI0Qnj9.png)

逻辑回归和不带核函数的`svm`很相似。

1. 当特征相对训练样本很多是，则使用逻辑回归或者不带核函数的`svm`
1. 当特征很少，且训练样本适中，则使用高斯核的`svm`
1. 当特征很少，且训练样本很多，则增加特征，使用逻辑回归或者不带核函数的`svm`
1. 良好的神经网络模型通常会得到较好的分类，但是训练会比较慢

#### 14.04 等式约束
假设有自变量`x`和`y`，给定约束条件`g(x,y)=c`，要求`f(x,y)`在约束`g`下的极值；

![等式约束](https://pic3.zhimg.com/v2-3cbdd04411f9b2f97e6d3939b45419ca_r.jpg)