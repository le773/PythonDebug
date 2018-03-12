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
向量a在向量b上投影的积,与余弦成正比;表示在相同方向上的相识度.

### 07 线性结合
### 08 内核
1. 
![径向基核](http://img.my.csdn.net/uploads/201304/03/1364958259_8460.jpg)
x1,x2越相近, K(x1,x2)越接近于1

2. ![多项式核](https://i.imgur.com/Daxm0UQ.png)
<x1,x2>:x1,x2的内积
k(x1,x2):x1 x2的相似性

3. 线性核
![线性核](http://img.my.csdn.net/uploads/201304/03/1364958354_7262.jpg)
实际上就是原始空间中的内积.

4. 双曲正切
![双曲正切](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D177/sign=adf59bb1f51fbe09185ec7135c610c30/96dda144ad345982648941550bf431adcaef84f2.jpg)


### 09 补充
假设有自变量x和y，给定约束条件g(x,y)=c，要求f(x,y)在约束g下的极值；
![拉格朗日乘子法](https://pic1.zhimg.com/359cdc26e15205e66204bce2b33e4535_r.jpg)
