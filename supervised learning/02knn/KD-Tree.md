### 1.0 KD-Tree的考虑
将KNN的搜索就可以被限制在空间的局部区域上，大大增加效率。

#### 1.1 如何决定每次根据哪个维度对子空间进行划分呢？
直观的来看，我们一般会选择轮流来(k维%n总维度)。先根据第一维，然后是第二维，然后第三...，那么到底轮流来行不行呢，这就要回到最开始我们为什么要研究选择哪一维进行划分的问题。我们研究Kd-Tree是为了优化在一堆数据中高频查找的速度，用树的形式，也是为了**尽快的缩小检索范围**，所以这个“比对维”就很关键，通常来说，更为分散的维度，我们就更容易的将其分开，是以这里我们通过求方差，用方差最大的维度来进行划分——这也就是最大方差法（`max invarince`）。

#### 1.2 如何选定根节点的比对数值呢？
选择何值未比对值，目的也是为了要加快检索速度。一般来说我们在构造一个二叉树的时候，当然是希望它是一棵尽量平衡的树，即左右子树中的结点个数相差不大。所以这里用当前维度的中值是比较合理的。

### 3.0 KD-Tree算法原理

给定一个构建于一个样本集的 kd 树，下面的算法可以寻找距离某个点 p 最近的 k 个样本。
```
(零)、设 L 为一个有 k 个空位的列表，用于保存已搜寻到的最近点。
(一)、根据 p 的坐标值和每个节点的切分向下搜索（也就是说，如果树的节点是按照 x<sub>r</sub>=a 进行切分，并且 p 的 r 坐标小于 a，则向左枝进行搜索；反之则走右枝）。
(二)、当达到一个底部节点时，将其标记为访问过。
    如果 L 里不足 k 个点，则将当前节点的特征坐标加入 L；
    如果 L 不为空并且当前节点的特征与 p 的距离小于 L 里最长的距离，则用当前特征替换掉 L 中离 p 最远的点。

(三)、如果当前节点不是整棵树最顶端节点，执行 (a)；反之，输出 L，算法完成。
a. 向上爬一个节点。
    如果当前（向上爬之后的）节点未曾被访问过，将其标记为被访问过，然后执行 (1) 和 (2)；
    如果当前节点被访问过，再次执行 (a)。

(1)此时如果 L 里不足 k 个点，则将节点特征加入 L；
       如果 L 中已满 k 个点，且当前节点与 p 的距离小于 L 里最长的距离，则用节点特征替换掉 L 中离最远的点。
(2) 计算 p 和当前节点切分线的距离。
    如果该距离大于等于 L 中距离 p 最远的距离并且 L 中已有 k 个点，则在切分线另一边不会有更近的点，执行(三)；
    如果该距离小于 L 中最远的距离或者 L 中不足 k 个点，则切分线另一边可能有更近的点，因此在当前节点的另一个枝从(一)开始执行。
```
### 4.1 KD-Tree树的建立
![kd-tree_2.png](https://i.imgur.com/j9xVAfZ.png)

### 4.2 判断轴是否与候选超球相交的方法
![kd-tree.png](https://i.imgur.com/qM3EQ29.png)

### 4.3 KD树搜索的复杂度
当实例随机分布的时候，搜索的复杂度为log(N)，N为实例的个数，KD树更加适用于实例数量远大于空间维度的KNN搜索，如果实例的空间维度与实例个数差不多时，它的效率基于等于线性扫描。

### 4.4 Kd-Tree和BST(二叉搜索树)的区别
BST的每个节点存储的是值，Kd-Tree的根节点和中间节点存储的是对某个维度的划分信息，只有叶节点里才是存储的值。

参考：

[图解 kd 树算法之详细篇](https://www.joinquant.com/post/2843)

[KNN算法中KD树的应用](http://kubicode.me/2015/10/12/Machine%20Learning/KDTree-In-KNN/)

[kd tree算法](http://www.cnblogs.com/eyeszjwang/articles/2429382.html)

[详解Kd-Tree](https://blog.csdn.net/qing101hua/article/details/53228668)

[Kd-Tree算法原理简析](https://blog.csdn.net/u012423865/article/details/77488920)