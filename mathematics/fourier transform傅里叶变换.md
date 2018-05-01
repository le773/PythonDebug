### 傅立叶变换定义
傅立叶变换，表示能将满足一定条件的某个函数表示成三角函数（正弦和/或余弦函数）或者它们的积分的线性组合。

### 形象解释傅里叶变换是什么？

下图，青色的波形是粉色和黄色波形的组合。

![fourier_1.png](https://i.imgur.com/jsLX2so.png)


如何分解途中黄色的波形呢？

![fourier_2.png](https://i.imgur.com/Wwrrviv.png)

首先，简单考虑只有一个频率的信号。

想象一个转到的向量，在任意时刻下它的长度等于这个时刻的图像高度，每过两秒，这个向量就转动一整圈。

![fourier_3.png](https://i.imgur.com/4n0NolZ.png)

现在有两个信号：缠绕频率 信号频率

![fourier_4.png](https://i.imgur.com/vqX1Jop.png)

所有低处的点恰好落在圆左侧，所有高处点恰好落在圆右侧。


----------

缠绕频率如何影响缠绕图像的质心

每个缠绕频率对应的质心位置的横坐标

![fourier_5.png](https://i.imgur.com/Fa7pA6b.png)

调整信号频率的坐标后：

![fourier_6.png](https://i.imgur.com/4UKKRRr.png)

上图可以看出，缠绕频率 信号频率 相等时，出现一个尖峰。

将上述情形推广到频率为2Hz和3Hz的混合模型上

![fourier_7.png](https://i.imgur.com/neNAzTe.png)

上图，在每秒2圈(每秒3圈)的时候图像整齐的排列起来

信号相加与近傅里叶变换

![fourier_8.png](https://i.imgur.com/pzroXHL.png)

对单纯频率的转换除了在其频率附近会出现一个尖峰以外，其它地方几乎是0，所以在将两个单纯频率加起来以后，转换后的图像就在输入进的频率处出现小尖峰。

----------

举例：音频编辑

![fourier_9.png](https://i.imgur.com/hmGw4Za.png)

上图，尖峰就是需要被去除的噪音。


----------

欧拉公式

![fourier_10.png](https://i.imgur.com/cgfTpdk.png)

![fourier_11.png](https://i.imgur.com/egvMOW2.png)

g(t):描述信号强度和时间关系的函数

e<sup>-2πift</sup>:t时间内划过弧度的长度

g(t) * e<sup>-2πift</sup>:旋转复数依照函数值大小被缩放了。

跟踪质心：

![fourier_12.png](https://i.imgur.com/cAqbikY.png)

![fourier_13.png](https://i.imgur.com/Y3d6nFL.png)

取极限时，就是对函数做积分，再除以时间区间的长度。


![fourier_14.png](https://i.imgur.com/lmVEAUq.png)

傅里叶变换就是在t2-t1时间内，质心位置的倍增。如果t2-t1很大，那这个频率的傅里叶变换的模长就被放得很大。

![fourier_15.png](https://i.imgur.com/E10u5Zt.png)

ghat(f)的输出值是一个复数，也就是在2维平面上的一个点，它对应原信号中某一频率的强度。
傅里叶变换只是这一函数的实部，也就是x轴坐标。

![fourier_16.png](https://i.imgur.com/P7Tzp3m.png)

在傅里叶变换中，积分通常是负无穷到正无穷，其含义是：...

傅里叶变换的另一种形式：

![fourier_17.png](https://i.imgur.com/RIGNU5I.png)

其中：

![fourier_18.png](https://i.imgur.com/P3V0Uag.png)

解读一下：

![fourier_19.png](https://i.imgur.com/tRZSBNK.png)

参考：

1. [如何直观形象、生动有趣地给文科学生介绍傅里叶变换？](https://www.zhihu.com/question/19991026/answer/252715189 "如何直观形象、生动有趣地给文科学生介绍傅里叶变换？")

1. [如何理解傅里叶变换公式？](https://www.zhihu.com/question/19714540/answer/334686351 "如何理解傅里叶变换公式？")

1. [形象展示傅里叶变换](https://www.bilibili.com/video/av19141078 "形象展示傅里叶变换")