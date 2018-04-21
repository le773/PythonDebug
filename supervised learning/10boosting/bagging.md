### 01 随机森林

1. 数据的随机性化
使得随机森林中的决策树更普遍化一点，适合更多的场景。

2. 待选特征的随机化
a. 子树从所有的待选特征中随机选取一定的特征。
b. 在选取的特征中选取最优的特征。


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
