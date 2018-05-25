### 1.0 熵的概念
#### 1.1 香农熵 Shannon entropy
信息熵（又叫香农熵）反映了一个系统的无序化（有序化）程度，一个系统越有序，信息熵就越低，反之就越高。

如果一个随机变量 X 的可能取值为 X={x1,x2,…,xn}，对应的概率为 p(X=xi)，则随机变量 X 的信息熵为：

![shannon_entropy_1.png](https://i.imgur.com/YnDHHvv.png)

#### 1.2 交叉熵 
用于衡量计算出的概率分布与真实的概率分布之间的差异。p表示真实标记的分布，q则为训练后的模型的预测标记分布。

![cross_entropy_1.png](https://i.imgur.com/N0xPQ0p.png)

#### 1.3 相对熵 relative entropy
所谓相对，自然在两个随机变量之间。又称互熵，Kullback–Leibler divergence（K-L 散度）等，用于衡量两个概率分布之间的差异。差异越大则相对熵越大，差异越小则相对熵越小，特别地，若2者相同则熵为0。注意，KL散度的非对称性。

设p(x)和q(x)是X取值的两个概率分布，则p对q的相对熵为： 

![relative_entropy_2.png](https://i.imgur.com/pSSscDX.png)

### 1.4 联合熵
![union_entropy.png](https://i.imgur.com/HEnL8rY.png)

(X,Y)在一起时的不确定性度量。
### 1.5 条件熵
![conditional entropy.png](https://i.imgur.com/egbHya3.png)

- X确定时，Y的不确定性度量
- 在X发生是前提下，Y发生新带来的熵

[如何通俗的解释交叉熵与相对熵?](https://www.zhihu.com/question/41252833 "如何通俗的解释交叉熵与相对熵?")