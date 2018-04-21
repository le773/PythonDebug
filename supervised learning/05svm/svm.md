### 00 svm

### 02 超平面的方程
margin:越大越好
二次规范:奇偶
hyperplanes:超平面
![超平面公式](http://img.blog.csdn.net/20131107201104906)
- w:代表平面参数
- b:离开原点的因素
- y:分类标签

### 03 平面之间的距离

#### 03.01 函数间隔 几何间隔

- 分隔超平面函数间距: \(y(x)=w^Tx+b\)
- 分类决策函数： \(f(x)=sign(w^Tx+b)\) (sign表示>0为1，<0为-1，=0为0)
- 点到超平面的几何间距: \(d(x)=(w^Tx+b)/||w||\) （||w||表示w矩阵的二范数=> \(\sqrt{w^T*w}\), 点到超平面的距离也是类似的）
- 二范数指矩阵A的2范数，就是A的转置共轭矩阵与矩阵A的积的最大特征根的平方根值，是指**空间上两个向量矩阵的直线距离**。

样本点(x_{i}, y_{i})与超平面(w, b)之间的函数间隔定义为![函数间隔](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bi%7D+%3D+y_%7Bi%7D+%28w%5Ccdot+x_%7Bi%7D+%2B+b%29++)；

几何间隔的定义：![几何间隔](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bi%7D+%3D+y_%7Bi%7D+%28%5Cfrac%7Bw%7D%7B%7C%7Cw%7C%7C%7D%5Ccdot+x_%7Bi%7D+%2B+%5Cfrac%7Bb%7D%7B%7C%7Cw%7C%7C%7D%29++)

![点到直线的距离](https://i.imgur.com/TDPTZGe.jpg)

**函数间隔**决定了数据点被分为某一类的确信度，就是上述距离公式中的分子，即未归一化的距离。
**几何间隔**实际上就是点到（超）平面的距离。
两者是一个||w||的线性关系。

#### 03.02 平面之间的距离 
如何求得一个几何间隔最大的分离超平面，即最大间隔分离超平面。具体地，这个问题可以表示为下面的约束最优化问题：
![最大间隔分离超平面](http://ww4.sinaimg.cn/large/6cbb8645gw1ewo70dte5nj209h02t3yf.jpg)

考虑几何间隔和函数间隔的关系式![几何间隔和函数间隔的关系式](http://ww1.sinaimg.cn/large/6cbb8645gw1ewo71ap0m0j202b01jmwx.jpg)，问题可以改写为：
![问题转换1](http://ww1.sinaimg.cn/large/6cbb8645gw1ewo71tvyhyj208002fq2t.jpg)

也即：
![超平面间的距离](https://i.imgur.com/Ez7dl4l.png)

#### 03.03 转化为凸二次规划

最大化`2/(w.T)`转化为约束最优化问题![凸二次规划](http://ww4.sinaimg.cn/large/6cbb8645gw1ewo7sn30ngj208m02ct8l.jpg)；平方项保证了表达式的单调性，它会放大结果但不会改变表达式的顺序。

#### 03.04 最大间隔法
![最大间隔法](http://ww3.sinaimg.cn/large/6cbb8645gw1ewsf9syt2bj20hf09lwfl.jpg)
最大间隔分离超平面的存在唯一性。

#### 03.05 支持向量
![支持向量](http://ww4.sinaimg.cn/large/6cbb8645gw1ewuxyftvpcj207s0673ym.jpg)

H1和H2平行，并且没有实例点落在它们中间。在H1和H2之间形成一条长带，分离超平面与它们平行且位于它们中央。长带的宽度，即H1和H2之间的距离称为间隔（margin)。间隔依赖于分离超平面的法向量w，等于![函数间隔](http://ww4.sinaimg.cn/large/6cbb8645gw1ewuy17xw1mj201101d741.jpg)。H1和H2称为间隔边界。

#### 03.06 拉格朗日函数
对每一个不等式约束![不等式约束](http://ww1.sinaimg.cn/large/6cbb8645gw1ewxghcdyupj208j00xweb.jpg)引进拉格朗日乘子，定义拉格朗日函数：

![拉格朗日函数](http://ww1.sinaimg.cn/large/6cbb8645gw1ewxiopiocuj209f01lgli.jpg)

其中，![拉格朗日乘子向量](http://ww4.sinaimg.cn/large/6cbb8645gw1ewxiwdf1kwj204100p742.jpg)为拉格朗日乘子向量。

经过一系列求解，得：
**w,b的解**：
![w,b的解](http://ww1.sinaimg.cn/large/6cbb8645gw1ewzkwec82kj204y02udfp.jpg)

分离超平面：
![分离超平面](http://ww2.sinaimg.cn/large/6cbb8645gw1ewzraq6n7gj204l01b3yb.jpg)

分类决策函数：
![分类决策函数](http://ww2.sinaimg.cn/large/6cbb8645gw1ewzrhgniqvj206j01jmx0.jpg)


#### 03.07 线性可分支持向量机学习算法

![线性可分支持向量机学习算法](http://ww2.sinaimg.cn/large/6cbb8645gw1ex5mbzp8x7j2080045jre.jpg)
线性可分训练集![线性可分训练集](http://ww3.sinaimg.cn/large/6cbb8645gw1ewzrwiwkw0j206g00mmwz.jpg)，其中![参数1](http://ww2.sinaimg.cn/large/6cbb8645gw1ewzs0pozfcj203l00p3ya.jpg)，![参数2](http://ww2.sinaimg.cn/large/6cbb8645gw1ewzs0q8908j205o00o3yb.jpg)


### 04 平面之间的距离 优化
![平面距离优化](https://i.imgur.com/BmK2un1.jpg)
- 几乎所有的数据都不那么干净, 通过引入松弛变量来 `允许数据点可以处于分隔面错误的一侧`。
- 约束条件： \(C>=a>=0\) 并且 \(\sum_{i=1}^{m} a_i·label_i=0\)

1. 松弛变量 表示 松弛变量
2. 常量C是 惩罚因子, 表示离群点的权重;
C值越大，表示离群点影响越大，就越容易过度拟合；反之有可能欠拟合。
3. 目标函数控制了离群点的数目和程度，使大部分样本点仍然遵守限制条件。

###05 序列最小优化(Sequential Minimal Optimization, SMO)

SMO目标：求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面。
SMO思想：是将大优化问题分解为多个小优化问题来求解的。
SMO原理：每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个。
a. 这里指的合适必须要符合一定的条件
  a1.这两个 alpha 必须要在间隔边界之外
  a2.这两个 alpha 还没有进行过区间化处理或者不在边界上。
b. 之所以要同时改变2个 alpha；原因是我们有一个约束条件： \(\sum_{i=1}^{m} a_i·label_i=0\)；如果只是修改一个 alpha，很可能导致约束条件失效。

代码流程
```
创建一个 alpha 向量并将其初始化为0向量
当迭代次数小于最大迭代次数时(外循环)
    对数据集中的每个数据向量(内循环)：
        如果该数据向量可以被优化
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
```

### 06 最佳分离器
注意:
距离边界很远的点不能用来定义决策边界的轮廓.因为对结果没有影响.

找出所有的点对,弄清楚哪些点是重要的,能够影响决策边界的定义,然后思考从其输出标签的角度,它们如何彼此相关.

点积:
![点积](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D268/sign=d3491000c9fc1e17f9bf8b377291f67c/63d9f2d3572c11df390c1835652762d0f603c2c3.jpg)
向量`a`在向量`b`上投影的积,与余弦成正比;表示在相同方向上的相识度.

### 07 线性结合

### 08 内核预览
1. 
![径向基核](http://img.my.csdn.net/uploads/201304/03/1364958259_8460.jpg)

`x1`,`x2`越相近, `K(x1,x2)`越接近于1

2. ![多项式核](https://i.imgur.com/Daxm0UQ.png)

`<x1,x2>:x1,x2`的内积
`k(x1,x2):x1 x2`的相似性

3. 线性核

![线性核](http://img.my.csdn.net/uploads/201304/03/1364958354_7262.jpg)

实际上就是原始空间中的内积.

4. 双曲正切
![双曲正切](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D177/sign=adf59bb1f51fbe09185ec7135c610c30/96dda144ad345982648941550bf431adcaef84f2.jpg)

- “伽玛”参数实际上对 `SVM` 的“线性”核函数没有影响。核函数的重要参数是“`C`”, 
- **C越大**，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，**趋向于对训练集全分对**的情况，这样对训练集测试时准确率很高，但泛化能力弱。**C值小**，对误分类的惩罚减小，允许容错，将他们当成噪声点，**泛化能力较强**;
- `degree`:多项式`poly`函数的维度;
- `kernel` ：核函数，默认是`rbf`，可以是‘`linear`’, ‘`poly`’, ‘`rbf`’, ‘`sigmoid`’, ‘`precomputed`’ 

#### 08.02 RBF公式里面的sigma和gamma的关系

![rbf中的gamma](http://img.blog.csdn.net/20150606105930104?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

`gamma`是你选择径向基函数作为`kernel`后，该函数自带的一个参数。隐含地决定了数据**映射到新的特征空间后的分布**。

如果`gamma`设的太大，`σ`会很小，`σ`很小的高斯分布长得又高又瘦，会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让`σ`无穷小，则理论上，高斯核的`SVM`可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过训练；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。

#### 08.03 rbf的优势
建议首选`RBF`核函数进行高维投影，因为：

1. 能够实现非线性映射；（ 线性核函数可以证明是他的一个特例；`SIGMOID`核函数在某些参数上近似RBF的功能。）
2. 参数的数量影响模型的复杂程度，多项式核函数参数较多。
3. `the RBF kernel has less numerical difficulties.`

#### 08.04 核函数总结
`Linear Kernel， Polynomial Kernel， Gaussian Kernel`

##### 08.04.01 Linear Kernel：K(x, x') = xTx'
**优点**是：
`safe`（一般不太会`overfitting`，所以线性的永远是我们的首选方案）；
`fast`，可以直接使用`General SVM`的`QP`方法来求解，比较迅速；
`explainable`，可解释性较好，我们可以直接得到`w, b`，它们直接对应每个`feature`的权重。
**缺点**是：
restrict：如果是线性不可分的资料就不太适用了！
 

##### 08.04.02 Polynomial Kernel: K(x, x') = (ζ + γxTx')Q       
**优点**是：
我们可以通过控制`Q`的大小任意改变模型的复杂度，一定程度上解决线性不可分的问题；
**缺点**是：
含有三个参数，太多啦！

##### 08.04.03 Gaussian Kernel：K(x, x') = exp(-γ ||x - x'||2) 
**优点**是：
`powerful`：比线性的`kernel`更`powerful`；
`bounded`：比多项式核更好计算一点；
`one  parameter only`：只有一个参数
**缺点**是：
`mysterious`：与线性核相反的是，可解释性比较差（先将原始数据映射到一个无限维度中，然后找一个胖胖的边界，将所有的数据点分隔开？）
`too powerful！`如果选择了太大的`γ`，`SVM`希望将所有的数据都分开，将会导致产生太过复杂的模型而`overfitting`。
###### 总结
所以在实际应用中，一般是先使用线性的`kernel`，如果效果不好再使用`gaussian kernel`（小的`γ`）和多项式`kernel`(小的`Q`)。

[svm详解](https://www.cnblogs.com/little-YTMM/p/5547642.html "svm详解")

### 09 补充
假设有自变量x和y，给定约束条件g(x,y)=c，要求f(x,y)在约束g下的极值；

![拉格朗日乘子法](https://pic1.zhimg.com/359cdc26e15205e66204bce2b33e4535_r.jpg)