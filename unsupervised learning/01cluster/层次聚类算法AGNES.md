### 1.0 层次聚类简介
层次聚类(`hierarchical clustering`)试图在不同层次对数据集进行划分，从而形成树形的聚类结构。数据集的划分可采用"自底向上"或"自顶向下"。

### 2.0 AGNES 算法
#### 2.1 AGNES 简介
AGNES是一种采用自底向上聚合策略的层次聚类算法，它先将数据集中的每个样本看作是一个初始聚类簇，然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，该过程不断重复，直至达到预设的聚类簇个数。

给定聚类簇C<sub>i</sub>与C<sub>j</sub>可通过下面的式子来计算距离：

![agnes_1.png](https://i.imgur.com/XZl6Qxh.png)

#### 2.2 Agnes算法伪代码

![agnes_2.png](https://i.imgur.com/ZAJiS9Y.png)

1. 先对仅含一个样本的初始聚类簇和相应的距离矩阵进行初始化；
2. 然后不断合并距离最近的聚类簇，并对合并得到的聚类簇的距离矩阵进行更新
3. 上述过程1，2不断重复，直到达到预设的聚类簇数。

#### 2.3 Agnes算法树状图

![agnes_3.png](https://i.imgur.com/H9vP5Px.png)
