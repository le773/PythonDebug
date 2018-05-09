![RandomForest.jpg](https://i.imgur.com/DeN3Hux.jpg)

### 01 随机森林
```
1.假如有N个样本,则有回放的随机选择N个样本（每次随机选择一个样本，然后返回继续选择）。
  这选择好了的N个样本用来训练一个决策树，作为决策树根节点处的样本。
2.当每个样本有M个属性时，在决策树的每个节点需要分裂时，随机从这M个属性中选取出m个属性，满足条件m<<M。
  然后从这m个属性中采用某种策略（如信息增益）来选择一个属性，作为该节点的分裂属性。
3.决策树形成过程中，每个节点都要按照步骤2来分裂
（很容易理解，如果下一次该节点选出来的那一个属性是刚刚父节点分裂时用过的属性，则该节点已经达到了叶子节点，无需继续分裂）。
  一直到不能再分裂为止，注意整个决策树形成过程中没有剪枝。
4.按步骤1-3建立大量决策树，如此形成随机森林。
```

1. 数据的随机性化
使得随机森林中的决策树更普遍化一点，适合更多的场景。

2. 待选特征的随机化
a. 子树从所有的待选特征中随机选取一定的特征。
b. 在选取的特征中选取最优的特征。

通过以上两点，使得最终集成的泛化性可通过个体学习器之间的差异度的增加而进一步提升。
### 02.01 RandomForestClassifier分类器

```
# GridSearchCV + AdaBoostClassifier
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

# TODO：初始化分类器
clf = RandomForestClassifier()
print 'clf'

# TODO：创建你希望调节的参数列表
# parameters_adaboost = {'n_estimators': [10, 20, 30, 40, 60, 100], 'algorithm' : ['SAMME', 'SAMME.R'], 'random_state': [40]}
parameter_grid = {
    'max_features': [0.5, 1.],
    'max_depth': [3., 5., 7.],
    'n_estimators': [10, 20, 30, 40, 60, 100]
}


# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, parameter_grid, scoring=scorer,cv=None)
print 'GridSearchCV'

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
t0 = time()
grid_obj.fit(X_train[:2000], y_train[:2000])
print 'fit time:', time() - t0
# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train[:1000], y_train[:1000])).predict(X_val)
print 'predict time:', time() - t1
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
```

### 02.02 提取特征重要性
```
# TODO：导入一个有'feature_importances_'的监督学习模型
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()

# TODO：在训练集上训练一个监督学习模型
model = clf.fit(X_train, y_train)

# TODO： 提取特征重要性
importances = model.feature_importances_

# 获取列名
indices = np.argsort(importances)[::-1]
columns = X_train.columns.values[indices[:5]]

# 获取数据
X_train[column_import].head(5)
```

### 3.0 RandomForest 随机森林预测伪代码

```
1.for n_trees in [1, 10, 20]:  # 理论上树是越多越好
    evaluate_algorithm(dataset, random_forest, n_folds=5, max_depth=20, min_size=1, sample_size=1.0, n_trees, n_features=15)
    1. 将数据集进行抽重抽样 n_folds 份，每份数据大小fold_size
    2. 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
    3. random_forest
        1. for i in range(n_trees):
            1. # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
            2. # 创建一个决策树
                1. get_split # 返回最优列和相关的信息
                    1. 找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]，以及分割完的数据 groups（left, right）
                        1. 从dataset中随机选取n_features 个特征
                        2. for index in features:
                               for row in dataset:
                                   1. # 根据特征和特征值分割，生成两个数据集
                                   2. 计算gini系数，# 左右两边的数量越一样，说明数据区分度不高，gini系数越大
                                   3. # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
                2. split # 对左右2边的数据 进行递归的调用，由于最优特征使用过，所以在后面进行使用的时候，就没有意义了，接下来需要找左右子树最佳分类
                    1. get_split ...
        2. bagging_predict # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
    4. # 计算随机森林的预测结果的正确率
```

传统决策树在选择划分属性时是在当前结点的属性集合(假定有d个属性)中选择一个最优属性；在RF中，对基决策树的每个节点，先从该结点的属性集合中随机选择一个包含k个属性的子集，然后再从这个子集中选择一个最优属性用于划分。

k控制了随机性的引入程度：若令k=d，则基决策树与传统决策树相同；若k=1，则是随机选择一个属性用于划分；一般情况下k=log<sub>2</sub>d。

上面代码中，features相当于k。

### 03.02 不同规模集成及其基学习器所对应的分类边界
![boosting_9.png](https://i.imgur.com/O8VfaA3.png)


### 4.0 结合策略
- 平均法
1.简单平均法

![bagging_1.png](https://i.imgur.com/AfJFyzE.png)

2.加权平均法

![bagging_2.png](https://i.imgur.com/yi1IXDJ.png)

w<sub>i</sub>是个体学习器h<sub>i</sub>的权重。

- 投票法
1.绝对多数投票法

![bagging_3.png](https://i.imgur.com/TxWLyKK.png)

即若某标记得票数过半数，则预测为该标记；否则拒绝预测。

2.相对多数投票法

![bagging_4.png](https://i.imgur.com/J0T5Haq.png)

即预测为得票最多的标记，若同时有多个标记获最高票，则随机选择一个。

3.加权投票法

![bagging_5.png](https://i.imgur.com/GemCrv9.png)

类似加权平均法