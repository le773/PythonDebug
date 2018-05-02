### 第二章 矩阵代数
##### 非奇异矩阵
若n阶方阵A的行列式不为零，即 |A|≠0，则称A为非奇异矩阵或满秩矩阵，

否则称A为奇异矩阵或降秩矩阵。
#### 2.4 分块矩阵
##### 2.4.1 分开矩阵的乘法
![matrix_1](https://i.imgur.com/k2CdxhG.png)

#### 2.5 矩阵因式分解
##### 2.5.1.1 LU分解
![matrix_LU_1](https://i.imgur.com/MOfHr61.png)

其中A是m*n矩阵，L是m*m矩阵，U是m*p矩阵；LU是A的分解

##### 2.5.1.2 LU分解的算法
![matrix_LU_2](https://i.imgur.com/A0hjAT1.png)

1. 首先，求出LU；
2. 然后，采用高斯消元法求y,x;

###### 例子
![matrix_LU_3.png](https://i.imgur.com/qoD2Gqd.png)

#### 2.5 维数与秩
矩阵A的秩是A的列空间的维数。

![matrix_2.png](https://i.imgur.com/5ZAQtHo.png)

- 矩阵A的秩是主元列的数量；
- 由主元列组成最大线性无关组；
###### 主元的定义
线性代数里面的**主元**，是指将一个矩阵A通过初等变换（包括初等行变换和列变换）化为规范阶梯型矩阵B后，矩阵B中每行从左往右，第一个非零的元素必定是1，这个1就是主元，所有主元的组合就是主元列。

##### 秩定理
如果矩阵A有n列，则rank(A) + dim NulA = n
非主元列对应于Ax=0中的自由变量。

### 第三章 矩阵代数
#### 3.2 行列式的性质
###### 3.2.1 行列式的计算1
按A的第一行的余因子展开式：

![matrix_4.png](https://i.imgur.com/Rhr2oOt.png)

Cij为余子式。

###### 3.2.2 行列式的计算1
1. 对行提取因子(行首归一化)
2. 然后高斯消元得到U型矩阵

![matrix_5.png](https://i.imgur.com/UHXEOYc.png)

- 定理4：方阵A是可逆的当且仅当det(A) != 0 
- 定理5：若A为一个n*n的矩阵，则det(A.T) = det.A
- 定理6：若A和B均为n*n的矩阵，则det(A*B)=(detA)*det(B)
一般而言，det(A+B) != det(A) + det(B)

#### 3.3 克拉默法则

![matrix_6.png](https://i.imgur.com/mRYChs5.png)

- 定理10：

![matrix_7.png](https://i.imgur.com/69qwF9M.png)

### 第六章 正交性和最小二乘法
#### 6.4 格拉姆-施密特方法
![matrix_8.png](https://i.imgur.com/wLdfHLU.png)

#### 6.5 最小二乘法
![matrix_9.png](https://i.imgur.com/idn4J51.png)

### 第七章 对称矩阵和二次型
#### 7.1 对称矩阵的对角化
![对称矩阵1](https://gss0.baidu.com/7Po3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=3045e06272c6a7efb973a020cdca8369/6a600c338744ebf8cd182182ddf9d72a6059a726.jpg)

- 定理1：如果A是对称矩阵，那么不同特征空间的任意两个特征向量是正交的。
- 定理2：一个n*n矩阵A可正交对角化的充分必要条件是A是对角矩阵。


