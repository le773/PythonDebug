### 01 序列最小优化(Sequential Minimal Optimization, SMO)
SMO目标：求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面。
SMO思想：是将大优化问题分解为多个小优化问题来求解的。
SMO原理：每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个。
```
a. 这里指的合适必须要符合一定的条件
  a1.这两个 alpha 必须要在间隔边界之外
  a2.这两个 alpha 还没有进行过区间化处理或者不在边界上。
```
![smo_1.png](https://i.imgur.com/AQiVFd8.png)

### 02 SMO伪代码
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

[SVM学习总结（一）如何学习SVM](https://blog.csdn.net/u010484388/article/details/54317837)

[SVM学习总结（二）SVM算法流程图](https://blog.csdn.net/u010484388/article/details/54317921)

[SVM学习总结（三）SMO算法流程图及注释源码](https://blog.csdn.net/u010484388/article/details/54318053)

[刘建平:支持向量机原理(一) 线性支持向量机](http://www.cnblogs.com/pinard/p/6097604.html)