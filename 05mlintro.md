### 01 机器学习

强化学习:延时反馈
监督学习:函数逼近 泛化函数

### 02 测试集的好处
1. 检查过拟合
2. 评估分类器或回归在独立数据集上的性能

#### 02.01 针对sklearn中K折的实用建议
不应该用A类训练模型，然后预测B类


### 03 训练模型
**居中趋势测量**：均值、中值、众数。
**数据的离散性**：四分位距法、异常值、标准偏差、贝塞尔修正。

### 04 检测错误
**过拟合**：太过具体，训练集表现很好，倾向于记住非学习
较低的训练误差，较高的测试误差
**欠拟合**：较高的训练误差，测试误差

#### 04.01 K折交叉验证
K折交叉验证是在优化模型时，将数据分为K等份，其中K-1份作为训练数据，1份作为测试数据。随后做交叉验证的时候，每次采用数据的其中1份作为测试集，计算模型的准确率。最终可以计算出K个准确率，利用K各准确率的平均值作为模型的准确度衡量。
实例
```
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```
`parameters`参数字典以及他们可取的值。在这种情况下，他们在尝试找到 kernel（可能的选择为 'linear' 和 'rbf' ）和 C（可能的选择为1和10）的最佳组合。这时，会自动生成一个不同（kernel、C）参数值组成的“网格”, 各组合均用于训练 SVM，并使用交叉验证对表现进行评估。


**C类**似于正则化中1λ1λ的作用。C越大，拟合非线性的能力越强。 
large C: High Variance
small C: High Bias

```
clf.fit(iris.data, iris.target)
```
第二个不可思议之处。 拟合函数现在尝试了所有的参数组合，并返回一个合适的分类器，自动调整至最佳参数组合。现在您便可通过`clf.best_params_`来获得参数值。

#### 04.02 网格搜索
网格搜索是用在选择模型参数的过程中。尝试所有的模型参数进行拟合，从中搜索到最佳的模型参数。之所以成为网格搜索，因为是在参数所建立的多维网格中找到最佳的参数点。

#### 04.03 K折交叉验证与网格搜索
网格搜索不使用交叉验证，使训练速度更快，但可能难以得到最优的模型参数；交叉验证对每一个参数组合得出的评分更为准确和鲁棒，提高评估的稳定性。

#### 04.04 学习曲线
![学习曲线](https://i.imgur.com/vFRX1dP.png)
- 欠拟合，随着训练集增加，训练误差和测试误差接近。
- 良好拟合，随着训练集增加，训练误差和测试误差接近，此时误差较欠拟合低。
- 过拟合，随着训练集增加，训练误差和测试误差接近，但是训练误差和测试误差相差较欠拟合大一些。交叉误差始终不会太低。

#### 04.05 模型复杂度
与学习曲线图形不同，**模型复杂度**图形呈现的是模型复杂度如何改变训练曲线和测试曲线，而不是呈现用来训练模型的数据点数量。一般趋势是，**随着模型增大，模型对固定的一组数据表现出更高的变化性**。

#### 04.06 使用RandomizeSearchCV来降低计算代价¶
- RandomizeSearchCV用于解决多个参数的搜索过程中计算代价过高的问题
- RandomizeSearchCV搜索参数中的一个子集，这样你可以控制计算代价 
[网格搜索来进行高效的参数调优](http://blog.csdn.net/jasonding1354/article/details/50562522 "网格搜索来进行高效的参数调优")

### 06 评估指标
#### 06.01 混淆矩阵
![混淆矩阵](https://i.imgur.com/K7TfktM.png)
#### 06.02 准确率
分类正确的(真阳真阴)/总树木
#### 06.03 准确率不适用的情形
可能会忽略一些点。
比如要找到信用卡不良的记录。

#### 06.05 精度和召回率
1. 对医疗来说，假阴不可接受，假阳可以。(高召回率)
因为健康的人可以识别为有病，但有病不能识别为健康。
2. 对垃圾邮件来说，假阴可以接受，假阳却不能。(高精度)
因为垃圾邮件可以接受，但正确邮件不能丢失。

#### 06.05 精度和召回率 计算

| - | Diagnosed sick | Diagnosed Healthy | 召回率recall |
| ---------- | :---------- : | ----------: |
| sick       | 正阳 | 假阴 | 正阳/(正阳+假阴)|
| healthy    | 假阳 | 正阴 |   |
|精确度precision| 正阳/(正阳+假阳)|            |

精确率
实际上非常简单，**精确率(precision)**是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是
![精度率](https://www.zhihu.com/equation?tex=P++%3D+%5Cfrac%7BTP%7D%7BTP%2BFP%7D)
而**召回率(recall)**是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。
![召回率](https://www.zhihu.com/equation?tex=R+%3D+%5Cfrac%7BTP%7D%7BTP%2BFN%7D)
其实就是分母不同，一个分母是预测为正的样本数，另一个是原来样本中所有的正样本数。
![精确率和召回率](https://pic1.zhimg.com/80/d701da76199148837cfed83901cea99e_hd.jpg)
在信息检索领域，精确率和召回率又被称为查准率和查全率，
查准率＝检索出的相关信息量 / 检索出的信息总量
查全率＝检索出的相关信息量 / 系统中的相关信息总量


#### 06.08 F1得分
![F1得分](https://i.imgur.com/3A8Y6mQ.png)

#### 06.09 Fβ得分
![fβ得分](https://i.imgur.com/B0cj2yH.png)
- β越小越偏向于精度，反之召回。
```
1对于宇宙飞船，我们不允许出现任何故障零件，可以检查本身能正常运转的零件。因此，这是一个高召回率模型，因此 β = 2。
2对于通知模型，因为是免费发送给客户，如果向更都的用户发送邮件也无害。但是也不能太过了，因为可能会惹怒用户。我们还希望找到尽可能多感兴趣的用户。因此，这个模型应该具有合适的精度和合适的召回率。β = 1 应该可行。
3对于促销材料模型，因为发送材料需要成本，我们不希望向不感兴趣的用户发送材料。因此是个高精度模型。β = 0.5 应该可行。
```
###### F-β得分的界限
![fβ得分的界限](https://i.imgur.com/xCinhF3.png)

#### 06.10 ROC曲线
![ROC曲线](https://i.imgur.com/UhjXxPI.png)
- roc曲线面积越接近于1,模型就越好

#### 06.11 决定系数 R2 
R2的数值范围从0至1，表示目标变量的预测值和实际值之间的相关程度平方的百分比。一个模型的R2 值为0还不如直接用平均值来预测效果好；而一个R2 值为1的模型则可以对目标变量进行完美的预测。从0至1之间的数值，则表示该模型中目标变量中有百分之多少能够用特征来解释。模型也可能出现负值的R2，这种情况下模型所做预测有时会比直接计算目标变量的平均值差很多。
```
def performance_metric2(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    y_mean = sum(y_true)/len(y_true)
    sst = sum(map(lambda x:(x-y_mean)**2, y_true))
    ssr = sum([(x-y)**2 for x, y in zip(y_true, y_predict)])
    score = 1- ssr/sst
    return score
```

![决定系数](https://i.imgur.com/Hq15eSP.png)

[决定系数](https://en.wikipedia.org/wiki/Coefficient_of_determination)


ssr:(训练模型得到的结果-平均值) *(训练模型得到的结果-平均值)之和

sst:(实际值-平均值) *(实际值-平均值)之和

sse:(实际值-训练模型得到的结果)*(实际值-训练模型得到的结果)之和


### 08 svm
- “伽玛”参数实际上对 SVM 的“线性”核函数没有影响。核函数的重要参数是“C”, 
- **C越大**，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，**趋向于对训练集全分对**的情况，这样对训练集测试时准确率很高，但泛化能力弱。**C值小**，对误分类的惩罚减小，允许容错，将他们当成噪声点，**泛化能力较强**;
- `degree`:多项式`poly`函数的维度;
- `kernel` ：核函数，默认是`rbf`，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 

0 – 线性：u'v

1 – 多项式：(gamma*u'*v + coef0)^degree

2 – RBF函数：exp(-gamma|u-v|^2)
也称高斯核函数

3 –sigmoid：tanh(gamma*u'*v + coef0)

#### 08.02 RBF公式里面的sigma和gamma的关系

![rbf中的gamma](http://img.blog.csdn.net/20150606105930104?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

gamma是你选择径向基函数作为kernel后，该函数自带的一个参数。隐含地决定了数据**映射到新的特征空间后的分布**。

如果gamma设的太大，![西格玛](http://img.blog.csdn.net/20150606110240260?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)会很小，![西格玛](http://img.blog.csdn.net/20150606110240260?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)很小的高斯分布长得又高又瘦，会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让![西格玛](http://img.blog.csdn.net/20150606110240260?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuZG9uZzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)无穷小，则理论上，高斯核的SVM可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过训练；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。

#### 08.03 rbf的优势
建议首选RBF核函数进行高维投影，因为：

1. 能够实现非线性映射；（ 线性核函数可以证明是他的一个特例；SIGMOID核函数在某些参数上近似RBF的功能。）
2. 参数的数量影响模型的复杂程度，多项式核函数参数较多。
3. the RBF kernel has less numerical difficulties.

#### 08.04 核函数总结
Linear Kernel， Polynomial Kernel， Gaussian Kernel

##### 08.04.01 Linear Kernel：K(x, x') = xTx'
**优点**是：
safe（一般不太会overfitting，所以线性的永远是我们的首选方案）；
fast，可以直接使用General SVM的QP方法来求解，比较迅速；
explainable，可解释性较好，我们可以直接得到w, b，它们直接对应每个feature的权重。
**缺点**是：
restrict：如果是线性不可分的资料就不太适用了！
 

##### 08.04.02 Polynomial Kernel: K(x, x') = (ζ + γxTx')Q       
**优点**是：
我们可以通过控制Q的大小任意改变模型的复杂度，一定程度上解决线性不可分的问题；
**缺点**是：
含有三个参数，太多啦！

##### 08.04.03 Gaussian Kernel：K(x, x') = exp(-γ ||x - x'||2) 
**优点**是：
powerful：比线性的kernel更powerful；
bounded：比多项式核更好计算一点；
one  parameter only：只有一个参数
**缺点**是：
mysterious：与线性核相反的是，可解释性比较差（先将原始数据映射到一个无限维度中，然后找一个胖胖的边界，将所有的数据点分隔开？）
too powerful！如果选择了太大的γ，SVM希望将所有的数据都分开，将会导致产生太过复杂的模型而overfitting。
###### 总结
所以在实际应用中，一般是先使用线性的kernel，如果效果不好再使用gaussian kernel（小的γ）和多项式kernel(小的Q)。
[svm详解](https://www.cnblogs.com/little-YTMM/p/5547642.html "svm详解")

### 09.01 RandomForestClassifier分类器

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

### 09.02 提取特征重要性
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