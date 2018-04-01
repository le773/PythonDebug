### 02 线性组合、张成的空间与基
向量v、w构成向量集合称为“张成的空间”(二维)

![linear_1](https://i.imgur.com/iHoOGuB.png)

向量v、w、u构成向量集合称为“张成的空间”(三维)

###### 线性无关
u与v、w线性无关
![linearly_independent_1](https://i.imgur.com/VGyf8C4.png)

**空间的一个基**的严格定义：张成该空间的一个线性无关向量的集合。

### 03 矩阵与线性变换
线性变换：原点保持不动，网格线保持平行且等距分布

###### 线性变换公式
![linear_transform_2](https://i.imgur.com/HHXs1U0.png)

###### 实例
![linear_transform_1](https://i.imgur.com/G6EUmgZ.png)

### 04 矩阵乘法与线性变换复合

两个**矩阵相乘**的意义：两个线性变换相继作用。

###### 线性变换相关概念
先旋转, 在剪切

![linear_transform_complex_1](https://i.imgur.com/22BIpb7.png)

###### 线性变换相关流程
![linear_transform_complex_2](https://i.imgur.com/lrjOgO1.png)

### 04 补充:三维空间中的线性变换

![linear_transform_complex_4](https://i.imgur.com/CsnOhHC.png)

### 05 行列式

![linear_transform_determinant_1](https://i.imgur.com/iQsQb9l.png)

向量i,j的**行列式为负**,代表**i,j组成的平面翻转**了。

三维空间：右手手指

![3D_vector_1](https://i.imgur.com/o1zs22u.png)

###### 行列式的计算
向量i,j行列式，即i,j构成平行四边形的面积.
![area_determinant_1](https://i.imgur.com/Zwt4C2c.png)

![area_determinant_2](https://i.imgur.com/c0BEtJS.png)

### 06 逆矩阵、列空间与零空间
**秩**：代表变换后空间的维数(列空间的维数)。

### 07 点积与对偶性
![linear_dot_1](https://i.imgur.com/pcMe7zW.png)

###### 点积与顺序无关
证明：假设w、v长度相等，他们相互的点积也相等。
假设：w延长两倍，则2*v * w = 2vw
同样假设v延长两倍，则点积为w*(2v在w上的投影，依旧为延长前的两倍)。
![linear_dot_2](https://i.imgur.com/UVhLvT8.png)

###### 点积相乘意义
两个点积相乘，就是将其中一个向量转化为线性变换。

![linear_dot_3](https://i.imgur.com/n89vquj.png)

### 08第一部分 叉积的标准介绍

![cross product_2](https://i.imgur.com/qbbCWxJ.png)

### 08第二部分 以线性变换的眼光看叉积

![cross product_3](https://i.imgur.com/0jPLCWQ.png)

### 09 基变换

![linear_transform_3](https://i.imgur.com/hwHGOLB.png)

###### A的逆*M*A

![linear_transform_4](https://i.imgur.com/aiKiKn4.png)

### 10 特征向量与特征值
**特征向量**:线性变换中,向量留在它所张成的空间里;

![linear_transform_6](https://i.imgur.com/l55RsHJ.png)

**特征值**:衡量特征向量在变换中拉伸或压缩的因子

![linear_transform_5](https://i.imgur.com/mgSNbIC.png)


![linear_transform_7](https://i.imgur.com/7kdTo8p.png)

#### 10.01 特征向量&特征值应用图解

![linear_transform_8](https://i.imgur.com/YTGshU9.png)

特征值出现负数的情况,一般对应于变换中的某种旋转.

#### 10.02 一个特征值对应多个特征向量

![linear_transform_9](https://i.imgur.com/HvscAl2.png)


### 11 抽象向量空间


