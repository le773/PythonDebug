### 01 线性回归
```
优点：结果易于理解，计算上不复杂。
缺点：对非线性的数据拟合不好。
适用于数据类型：数值型和标称型数据。
```
#### 01.01.01 最小二乘法(平方误差)
![linear_regression_1.png](https://i.imgur.com/nUKZePq.png)

#### 01.01.02 平方误差求解权重w
![linear_regression_2.png](https://i.imgur.com/U8Rhw3Q.png)

### 02 局部加权线性回归
线性回归的一个问题是有可能出现**欠拟合**现象，因为它求的是具有最小均方差的无偏估计。显而易见，如果模型欠拟合将不能取得最好的预测效果。所以有些方法允许在估计中引入一些偏差，从而降低预测的均方误差。

局部加权线性回归 工作原理:

1. 读入数据，将数据特征x、特征标签y存储在矩阵x、y中
1. 利用高斯核构造一个权重矩阵 W，对预测点附近的点施加权重
1. 验证 X^TWX 矩阵是否可逆
1. 使用最小二乘法求得 回归系数 w 的最佳估计


#### 02.01.01 最小化目标函数
![linear_regression_3.png](https://i.imgur.com/Lv1vmVZ.png)

#### 02.01.02 回归系数

![linear_regression_4.png](https://i.imgur.com/i51TtyS.png)

W是权重，what是回归系数。

### 03 岭回归 ridge regression

背景：解决样本少于特征

矩阵 矩阵X<sup>T</sup>X上加一个 λI 从而使得矩阵非奇异，进而能对 矩阵X<sup>T</sup>X + λI 求逆; 其中矩阵I是一个 n * n 的单位矩阵

#### 03.01.01 回归系数

![linear_regression_5.png](https://i.imgur.com/sCU8oAw.png)

通过引入 λ 来限制了所有 w 之和，通过引入该惩罚项，能够减少不重要的参数，这个技术在统计学中也叫作 缩减(`shrinkage`)。

### 04 套索方法 Lasso
#### 04.01.01 回归系数约束
L1约束

![linear_regression_6.png](https://i.imgur.com/3FBXiJY.png)

L2约束

![linear_regression_7.png](https://i.imgur.com/9WmxZBR.png)

### 05 前向逐步回归
#### 05.01 伪代码
```
数据标准化，使其分布满足 0 均值 和单位方差
在每轮迭代过程中: 
    设置当前最小误差 lowestError 为正无穷
    对每个特征:
        增大或缩小:
            改变一个系数得到一个新的 w
            计算新 w 下的误差
            如果误差 Error 小于当前最小误差 lowestError: 设置 Wbest 等于当前的 W
        将 W 设置为新的 Wbest
```
### 06 权衡偏差和方差

![linear_regression_8.png](https://i.imgur.com/BM5Si5B.png)

上面的曲面就是测试误差，下面的曲线是训练误差;
从下图开看，从左到右就表示了核逐渐减小的过程;


