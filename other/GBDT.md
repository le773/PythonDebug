对于regression问题，GradientBoost通过residual fitting的方式得到最佳的方向函数 g<sub>t</sub> 和步进长度 η。

AdaBoost算法所做的其实是在gradient descent上找到下降最快的方向和最大的步进长度。这里的方向就是 g<sub>t</sub> ，它是一个函数，而步进长度就是α<sub>t</sub>。

![GradientBoost](https://pic1.zhimg.com/80/v2-fcccd1080b1a781de7ff2affa63799bb_hd.jpg)

h(x<sub>n</sub>):是下一步前进的方向。
η：步进长度

aggregation的两个优势：feature transform和regularization
https://zhuanlan.zhihu.com/p/36681976


uniformly：一人一票
non-uniformly：每个人的投票权重不同
blending：选择的性能较好的一些矩gtgt，将它们进行整合、合并，来得到最佳的预测模型。



uniform blending的做法很简单，就是将所有的矩 g_t 求平均值

bootstrapping是统计学的一个工具，思想就是从已有数据集D中模拟出其他类似的样本 D<sub>t</sub>。

bootstrap:其实质是对观测信息进行再抽样，进而对总体的分布特性进行统计推断。

首先，Bootstrap通过重抽样，可以避免了Cross-Validation造成的样本减少问题，其次，Bootstrap也可以用于创造数据的随机性。比如，我们所熟知的随机森林算法第一步就是从原始训练数据集中，应用bootstrap方法有放回地随机抽取k个新的自助样本集，并由此构建k棵分类回归树。

Hybrid model structure. input features are transformed by means of boosted decision trees.the output of each individual tree is treated as a categorical input feature to a sparse linear classifier.Boosted decision trees prove to be very powerful feature transforms.

![gbdt](https://images2017.cnblogs.com/blog/666027/201710/666027-20171031154748527-1827071972.png)

待阅：https://www.cnblogs.com/ModifyRong/category/835572.html

机器学习系列------1. GBDT算法的原理
https://blog.csdn.net/u012684933/article/details/51088609

机器学习中的算法(1)-决策树模型组合之随机森林与GBDT

http://www.cnblogs.com/leftnoteasy/archive/2011/03/07/random-forest-and-gbdt.html

[Gradient Boost 算法流程分析](https://blog.csdn.net/w28971023/article/details/8133929)

c<sub>tj</sub>:拟合叶子节点
x<sub>i</sub>:样本
r<sub>ti</sub>:负梯度
R<sub>tj</sub>:叶子节点区域
c<sub>tj</sub>:最佳拟合值
f(x):学习器