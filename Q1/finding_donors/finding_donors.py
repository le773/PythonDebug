
# coding: utf-8

# # Python
# ## 项目2: 为*CharityML*寻找捐献者

# 欢迎来到机器学习工程师纳米学位的第二个项目！在此文件中，有些示例代码已经提供给你，但你还需要实现更多的功能让项目成功运行。除非有明确要求，你无须修改任何已给出的代码。以**'练习'**开始的标题表示接下来的代码部分中有你必须要实现的功能。每一部分都会有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示！
# 
# 除了实现代码外，你还必须回答一些与项目和你的实现有关的问题。每一个需要你回答的问题都会以**'问题 X'**为标题。请仔细阅读每个问题，并且在问题后的**'回答'**文字框中写出完整的答案。我们将根据你对问题的回答和撰写代码所实现的功能来对你提交的项目进行评分。
# >**提示：**Code 和 Markdown 区域可通过**Shift + Enter**快捷键运行。此外，Markdown可以通过双击进入编辑模式。

# ## 开始
# 
# 在这个项目中，你将使用1994年美国人口普查收集的数据，选用几个监督学习算法以准确地建模被调查者的收入。然后，你将根据初步结果从中选择出最佳的候选算法，并进一步优化该算法以最好地建模这些数据。你的目标是建立一个能够准确地预测被调查者年收入是否超过50000美元的模型。这种类型的任务会出现在那些依赖于捐款而存在的非营利性组织。了解人群的收入情况可以帮助一个非营利性的机构更好地了解他们要多大的捐赠，或是否他们应该接触这些人。虽然我们很难直接从公开的资源中推断出一个人的一般收入阶层，但是我们可以（也正是我们将要做的）从其他的一些公开的可获得的资源中获得一些特征从而推断出该值。
# 
# 这个项目的数据集来自[UCI机器学习知识库](https://archive.ics.uci.edu/ml/datasets/Census+Income)。这个数据集是由Ron Kohavi和Barry Becker在发表文章_"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_之后捐赠的，你可以在Ron Kohavi提供的[在线版本](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)中找到这个文章。我们在这里探索的数据集相比于原有的数据集有一些小小的改变，比如说移除了特征`'fnlwgt'` 以及一些遗失的或者是格式不正确的记录。

# ----
# ## 探索数据
# 运行下面的代码单元以载入需要的Python库并导入人口普查数据。注意数据集的最后一列`'income'`将是我们需要预测的列（表示被调查者的年收入会大于或者是最多50,000美元），人口普查数据中的每一列都将是关于被调查者的特征。

# In[1]:


# 检查你的Python版本
from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7来完成此项目')


# In[2]:


# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
import visuals as vs

# 为notebook提供更加漂亮的可视化
get_ipython().magic(u'matplotlib inline')

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))


# ### 练习：数据探索
# 首先我们对数据集进行一个粗略的探索，我们将看看每一个类别里会有多少被调查者？并且告诉我们这些里面多大比例是年收入大于50,000美元的。在下面的代码单元中，你将需要计算以下量：
# 
# - 总的记录数量，`'n_records'`
# - 年收入大于50,000美元的人数，`'n_greater_50k'`.
# - 年收入最多为50,000美元的人数 `'n_at_most_50k'`.
# - 年收入大于50,000美元的人所占的比例， `'greater_percent'`.
# 
# **提示：** 您可能需要查看上面的生成的表，以了解`'income'`条目的格式是什么样的。 

# In[3]:


# TODO：总的记录数
n_records = data['age'].shape[0]
n_records


# In[4]:


data['income'].describe()


# In[5]:


data['income'].unique()


# In[6]:


# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = data.query('income == ">50K"')['age'].count()
n_greater_50k


# In[7]:


n_lower_50k = data.query('income == "<=50K"')['age'].count()
n_lower_50k


# In[8]:


# TODO：被调查者收入大于$50,000所占的比例
n_greater_50k_scale = float(n_greater_50k)/n_records
print "A:{} B:{}".format(n_greater_50k, n_records)
n_greater_50k_scale


# In[9]:


# TODO：总的记录数
n_records = data['age'].count()

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = data.query('income == ">50K"')['age'].count()

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = data.query('income == "<=50K"')['age'].count()

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = float(n_greater_50k)/n_records

# 打印结果
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)


# ----
# ## 准备数据
# 在数据能够被作为输入提供给机器学习算法之前，它经常需要被清洗，格式化，和重新组织 - 这通常被叫做**预处理**。幸运的是，对于这个数据集，没有我们必须处理的无效或丢失的条目，然而，由于某一些特征存在的特性我们必须进行一定的调整。这个预处理都可以极大地帮助我们提升几乎所有的学习算法的结果和预测能力。
# 
# ### 获得特征和标签
# `income` 列是我们需要的标签，记录一个人的年收入是否高于50K。 因此我们应该把他从数据中剥离出来，单独存放。

# In[10]:


# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)


# In[11]:


data.head(1)


# ### 转换倾斜的连续特征
# 
# 一个数据集有时可能包含至少一个靠近某个数字的特征，但有时也会有一些相对来说存在极大值或者极小值的不平凡分布的的特征。算法对这种分布的数据会十分敏感，并且如果这种数据没有能够很好地规一化处理会使得算法表现不佳。在人口普查数据集的两个特征符合这个描述：'`capital-gain'`和`'capital-loss'`。
# 
# 运行下面的代码单元以创建一个关于这两个特征的条形图。请注意当前的值的范围和它们是如何分布的。

# In[12]:


# 可视化 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw)


# 对于高度倾斜分布的特征如`'capital-gain'`和`'capital-loss'`，常见的做法是对数据施加一个<a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">对数转换</a>，将数据转换成对数，这样非常大和非常小的值不会对学习算法产生负面的影响。并且使用对数变换显著降低了由于异常值所造成的数据范围异常。但是在应用这个变换时必须小心：因为0的对数是没有定义的，所以我们必须先将数据处理成一个比0稍微大一点的数以成功完成对数转换。
# 
# 运行下面的代码单元来执行数据的转换和可视化结果。再次，注意值的范围和它们是如何分布的。

# In[13]:


# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 可视化对数转换后 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw, transformed = True)


# ### 规一化数字特征
# 除了对于高度倾斜的特征施加转换，对数值特征施加一些形式的缩放通常会是一个好的习惯。在数据上面施加一个缩放并不会改变数据分布的形式（比如上面说的'capital-gain' or 'capital-loss'）；但是，规一化保证了每一个特征在使用监督学习器的时候能够被平等的对待。注意一旦使用了缩放，观察数据的原始形式不再具有它本来的意义了，就像下面的例子展示的。
# 
# 运行下面的代码单元来规一化每一个数字特征。我们将使用[`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来完成这个任务。

# In[14]:


from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
display(features_raw.head(n = 1))


# ### 练习：数据预处理
# 
# 从上面的**数据探索**中的表中，我们可以看到有几个属性的每一条记录都是非数字的。通常情况下，学习算法期望输入是数字的，这要求非数字的特征（称为类别变量）被转换。转换类别变量的一种流行的方法是使用**独热编码**方案。独热编码为每一个非数字特征的每一个可能的类别创建一个_“虚拟”_变量。例如，假设`someFeature`有三个可能的取值`A`，`B`或者`C`，。我们将把这个特征编码成`someFeature_A`, `someFeature_B`和`someFeature_C`.
# 
# | 特征X |                    | 特征X_A | 特征X_B | 特征X_C |
# | :-: |                            | :-: | :-: | :-: |
# |  B  |  | 0 | 1 | 0 |
# |  C  | ----> 独热编码 ----> | 0 | 0 | 1 |
# |  A  |  | 1 | 0 | 0 |
# 
# 此外，对于非数字的特征，我们需要将非数字的标签`'income'`转换成数值以保证学习算法能够正常工作。因为这个标签只有两种可能的类别（"<=50K"和">50K"），我们不必要使用独热编码，可以直接将他们编码分别成两个类`0`和`1`，在下面的代码单元中你将实现以下功能：
#  - 使用[`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies)对`'features_raw'`数据来施加一个独热编码。
#  - 将目标标签`'income_raw'`转换成数字项。
#    - 将"<=50K"转换成`0`；将">50K"转换成`1`。

# In[15]:


# 数字化
# pd.get_dummies(features_raw).head(4)


# In[16]:


# 数字化
# income = pd.get_dummies(income_raw)
# income.head(4)


# In[17]:


# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
# income = pd.get_dummies(income_raw)
income = income_raw.map({'<=50K':0, '>50K':1})
income.unique()

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# 移除下面一行的注释以观察编码的特征名字
print encoded


# ### 混洗和切分数据
# 现在所有的 _类别变量_ 已被转换成数值特征，而且所有的数值特征已被规一化。和我们一般情况下做的一样，我们现在将数据（包括特征和它们的标签）切分成训练和测试集。其中80%的数据将用于训练和20%的数据用于测试。然后再进一步把训练数据分为训练集和验证集，用来选择和优化模型。
# 
# 运行下面的代码单元来完成切分。

# In[18]:


# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

# 显示切分的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Validation set has {} samples.".format(X_val.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# In[19]:


X_train.head()


# In[20]:


X_val.head()


# In[21]:


y_train.head()


# In[22]:


y_val.head()


# ----
# ## 评价模型性能
# 在这一部分中，我们将尝试四种不同的算法，并确定哪一个能够最好地建模数据。四种算法包含一个*天真的预测器* 和三个你选择的监督学习器。

# ### 评价方法和朴素的预测器
# *CharityML*通过他们的研究人员知道被调查者的年收入大于\$50,000最有可能向他们捐款。因为这个原因*CharityML*对于准确预测谁能够获得\$50,000以上收入尤其有兴趣。这样看起来使用**准确率**作为评价模型的标准是合适的。另外，把*没有*收入大于\$50,000的人识别成年收入大于\$50,000对于*CharityML*来说是有害的，因为他想要找到的是有意愿捐款的用户。这样，我们期望的模型具有准确预测那些能够年收入大于\$50,000的能力比模型去**查全**这些被调查者*更重要*。我们能够使用**F-beta score**作为评价指标，这样能够同时考虑查准率和查全率：
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# 
# 尤其是，当 $\beta = 0.5$ 的时候更多的强调查准率，这叫做**F$_{0.5}$ score** （或者为了简单叫做F-score）。

# ### 问题 1 - 天真的预测器的性能
# 
# 通过查看收入超过和不超过 \$50,000 的人数，我们能发现多数被调查者年收入没有超过 \$50,000。如果我们简单地预测说*“这个人的收入没有超过 \$50,000”*，我们就可以得到一个 准确率超过 50% 的预测。这样我们甚至不用看数据就能做到一个准确率超过 50%。这样一个预测被称作是天真的。通常对数据使用一个*天真的预测器*是十分重要的，这样能够帮助建立一个模型表现是否好的基准。 使用下面的代码单元计算天真的预测器的相关性能。将你的计算结果赋值给`'accuracy'`, `‘precision’`, `‘recall’` 和 `'fscore'`，这些值会在后面被使用，请注意这里不能使用scikit-learn，你需要根据公式自己实现相关计算。
# 
# *如果我们选择一个无论什么情况都预测被调查者年收入大于 \$50,000 的模型，那么这个模型在**验证集上**的准确率，查准率，查全率和 F-score是多少？*  
# 

# In[23]:


y_val.head()


# In[24]:


y_val.shape[0]


# In[25]:


y_val[y_val == 1].shape[0]


# In[26]:


#不能使用scikit-learn，你需要根据公式自己实现相关计算。

total = y_val.shape[0]
test_right = y_val[y_val == 1].shape[0]

#TODO： 计算准确率
accuracy = 1.0 *  test_right/total

# TODO： 计算查准率 Precision
tp = test_right
precision = 1.0 * tp/total

# TODO： 计算查全率 Recall
recall = 1.0 * tp/tp

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
beta=0.5
fscore = (1 + beta * beta) * precision * recall /(beta * beta * precision + recall)

# 打印结果
print "Naive Predictor on validation data: \n     Accuracy score: {:.4f} \n     Precision: {:.4f} \n     Recall: {:.4f} \n     F-score: {:.4f}".format(accuracy, precision, recall, fscore)


# 准确率：0.2478 
# 查准率：0.2478
# 查全率：1.0000 
# F-score：0.2917

# ## 监督学习模型
# ### 问题 2 - 模型应用
# 
# 你能够在 [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) 中选择以下监督学习模型
# - 高斯朴素贝叶斯 (GaussianNB)
# - 决策树 (DecisionTree)
# - 集成方法 (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K近邻 (K Nearest Neighbors)
# - 随机梯度下降分类器 (SGDC)
# - 支撑向量机 (SVM)
# - Logistic回归（LogisticRegression）
# 
# 从上面的监督学习模型中选择三个适合我们这个问题的模型，并回答相应问题。

# ### 2.01 高斯朴素贝叶斯
# 
# **模型名称**
# 
# 回答：高斯朴素贝叶斯
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：垃圾邮件过滤中的应用
# 
# [遗传算法和贝叶斯模型在垃圾邮件过滤中的应用](http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=D663789_)
# 
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 对小规模的数据表现很好，能个处理多分类任务，适合增量式训练；
# - 面对孤立的噪声点，朴素贝叶斯分类器是健壮的；
# - 面对无关属性，NB是健壮的；
# 
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 需要计算先验概率；
# - 分类决策存在错误率；
# - 对输入数据的表达形式很敏感；
# - 输入变量必须都是条件独立的；
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：

# ###  2.02  决策树
# 
# **模型名称**
# 
# 回答：决策树
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：[决策树之银行行销预测应用分析](https://www.ibm.com/developerworks/cn/data/library/ba/ba-1412spssmodelerbank/)
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 计算简单，易于理解，可解释性强；
# - 比较适合处理有缺失属性的样本；
# - 能够处理不相关的特征；
# - 在相对短的时间内能够对大型数据源做出可行且效果良好的结果。
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 容易发生过拟合（随机森林可以很大程度上减少过拟合），剪枝是解决决策树过拟合并缩小决策树的方法之一；
# - 忽略了数据之间的相关性；
# - 对于那些各类别样本数量不一致的数据，在决策树当中,信息增益的结果偏向于那些具有更多数值的特征（只要是使用了信息增益，都有这个缺点，如RF）。
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：在相对短的时间内能够对此模型数据源做出可行且效果良好的结果。

# ### 2.03 随机森林
# 
# **模型名称**
# 
# 回答：随机森林
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：[随机森林机器学习算法在桥梁检测中的应用](http://xueshu.baidu.com/s?wd=paperuri%3A(08823165971a7095c22b3cf426a240e7)&filter=sc_long_sign&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_ks_para=q%3D%E6%94%B9%E8%BF%9B%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E7%AE%97%E6%B3%95%E5%9C%A8Android%E6%81%B6%E6%84%8F%E8%BD%AF%E4%BB%B6%E6%A3%80%E6%B5%8B%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8)
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 在当前的很多数据集上，相对其他算法有着很大的优势，表现良好
# - 它能够处理很高维度（feature很多）的数据，并且不用做特征选择
# - 在训练完后，它能够给出哪些feature比较重要
# - 在创建随机森林的时候，对generlization error使用的是无偏估计，模型泛化能力强
# - 训练速度快，容易做成并行化方法
# - 在训练过程中，能够检测到feature间的互相影响
# - 实现比较简单
# - 对于不平衡的数据集来说，它可以平衡误差。
# - 如果有很大一部分的特征遗失，仍可以维持准确度。
# 
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 随机森林已经被证明在某些噪音较大的分类或回归问题上会过拟合
# - 对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的。
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：能够处理高维度的数据(103个feature)，可以并行计算

# ### 2.04 K近邻
# 
# **模型名称**
# 
# 回答：K近邻
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：[k-近邻判别分析法在个人信用评估中的应用](http://xueshu.baidu.com/s?wd=paperuri%3A%28ec682393dfaaeb58cfe58e49d11edb2e%29&filter=sc_long_sign&sc_ks_para=q%3Dk-%E8%BF%91%E9%82%BB%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90%E6%B3%95%E5%9C%A8%E4%B8%AA%E4%BA%BA%E4%BF%A1%E7%94%A8%E8%AF%84%E4%BC%B0%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8&sc_us=2017396846961085426&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8)
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 理论成熟，思想简单，既可以用来做分类也可以用来做回归；
# - 可用于非线性分类，可以生成任意形状的决策边界；
# - 训练时间复杂度为O(n)，因为是消极的学习方法不需要建立模型；
# - 对数据没有假设，准确度高，对outlier不敏感；
# - 基于局部信息进行预测，正因为这样而局部分类决策，KNN（k很小时）对噪声非常敏感；
# 
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 计算量大（体现在距离计算上），因为需要逐个计算测试样例和训练样例之间的而相似度；
# - 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）效果差；
# - 需要大量内存；
# - 除非采用适当的邻近性度量k和数据预处理，否则最近邻分类器可能做出错误的预测
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：理论成熟。
# (不适合在本地电脑执行，笔者使用knn和网格搜索结合，训练2w+数据20min都没跑出结果...)

# ### 2.05 随机梯度下降分类器
# 
# **模型名称**
# 
# 回答：随机梯度下降分类器
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：图片识别
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 效率较高,易于实现
# - 适合特别大数据集
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 对特征缩放比较敏感
# - 需要大量的超参数(hyperparameters)
# - 每次更新可能并不会按照正确的方向进行，因此可以带来优化波动(扰动)
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：当前数据集训练集量比较大

# ### 2.06 SVM
# 
# **模型名称**
# 
# 回答：支持向量机
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：[基于支持向量机的网络入侵检测](http://xueshu.baidu.com/s?wd=paperuri%3A%284cf314cb42aa65bbb98dfff776f7a0e7%29&filter=sc_long_sign&sc_ks_para=q%3D%E5%9F%BA%E4%BA%8E%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%E7%9A%84%E7%BD%91%E7%BB%9C%E5%85%A5%E4%BE%B5%E6%A3%80%E6%B5%8B&sc_us=12189002413550318681&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8)
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 可以解决高维问题，即大型特征空间；
# - 能够处理非线性特征的相互作用；
# - 无需依赖整个数据；
# - 可以提高泛化能力；
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 当观测样本很多时，效率并不是很高，因为在大数据中，SVM时间复杂度为O(n^{3}) ；
# - 对非线性问题没有通用解决方案，有时候很难找到一个合适的核函数；
# - 在噪声过多的情况下，SVM容易造成过拟合
# - 类严重重叠时，SVM的表现也很差
# - 对缺失数据敏感；
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：适合解决高为问题，但是计算量很大

# ### 2.07 Logistic回归
# 
# **模型名称**
# 
# 回答：Logistic回归
# 
# 
# **描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**
# 
# 回答：
# 
# **这个模型的优势是什么？他什么情况下表现最好？**
# 
# 回答：
# ##### 优点
# 
# - 实现简单，广泛的应用于工业问题上；
# - 分类时计算量非常小，速度很快，存储资源低；
# - 便利的观测样本概率分数；
# - 对逻辑回归而言，多重共线性并不是问题，它可以结合L2正则化来解决该问题；
# 
# 
# **这个模型的缺点是什么？什么条件下它表现很差？**
# 
# ##### 缺点
# 
# - 当特征空间很大时，逻辑回归的性能不是很好；
# - 容易欠拟合，一般准确度不太高
# - 不能很好地处理大量多类特征或变量；
# - 只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；
# - 对于非线性特征，需要进行转换
# 
# 
# **根据我们当前数据集的特点，为什么这个模型适合这个问题。**
# 
# 回答：当前问题判断高低收入为二分类问题

# ### 练习 - 创建一个训练和预测的流水线
# 为了正确评估你选择的每一个模型的性能，创建一个能够帮助你快速有效地使用不同大小的训练集并在验证集上做预测的训练和验证的流水线是十分重要的。
# 你在这里实现的功能将会在接下来的部分中被用到。在下面的代码单元中，你将实现以下功能：
# 
#  - 从[`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)中导入`fbeta_score`和`accuracy_score`。
#  - 用训练集拟合学习器，并记录训练时间。
#  - 对训练集的前300个数据点和验证集进行预测并记录预测时间。
#  - 计算预测训练集的前300个数据点的准确率和F-score。
#  - 计算预测验证集的准确率和F-score。

# In[27]:


# TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_val, y_val): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''
    
    results = {}
    
    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # 获得程序开始时间
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
#     X_sample=X_train.sample(sample_size)
#     y_sample=y_train.ix[X_sample.index]
#     learner = learner.fit(X_sample,y_sample)
    
    end = time() # 获得程序结束时间
    
    # TODO：计算训练时间
    results['train_time'] = end - start
    
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train[:300])
    end = time() # 获得程序结束时间
    
    # TODO：计算预测用时
    results['pred_time'] = end - start
            
    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO：计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)
    
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, average=None, beta=0.5)
        
    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, average=None, beta=0.5)
       
    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
#     print results
    # 返回结果
    return results


# ### 练习：初始模型的评估
# 在下面的代码单元中，您将需要实现以下功能：             
# - 导入你在前面讨论的三个监督学习模型。             
# - 初始化三个模型并存储在`'clf_A'`，`'clf_B'`和`'clf_C'`中。
#   - 使用模型的默认参数值，在接下来的部分中你将需要对某一个模型的参数进行调整。             
#   - 设置`random_state`  (如果有这个参数)。       
# - 计算1%， 10%， 100%的训练数据分别对应多少个数据点，并将这些值存储在`'samples_1'`, `'samples_10'`, `'samples_100'`中
# 
# **注意：**取决于你选择的算法，下面实现的代码可能需要一些时间来运行！

# In[49]:


# TODO：从sklearn中导入三个监督学习模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# TODO：初始化三个模型
clf_A = DecisionTreeClassifier(random_state=0)
clf_B = SVC()
clf_C = LogisticRegression(random_state=10)
clf_D = GaussianNB()
clf_E = KNeighborsClassifier(n_neighbors=5)
clf_F = RandomForestClassifier()

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(0.01 * n_records)
samples_10 = int(0.1 * n_records)
samples_neg_1 = int(0.001 * n_records)
samples_100 = n_records

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_D, clf_E, clf_B, clf_C, clf_F]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
#     for i, samples in enumerate([samples_neg_1, samples_1, samples_10]):
    for i, samples in enumerate([samples_1, samples_10,samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 对选择的三个模型得到的评价结果进行可视化
def sub_dict(somedict,somekeys,default=None):  
    return dict([(k,somedict.get(k,default)) for k in somekeys])  

somekeys=(clf_B.__class__.__name__, clf_F.__class__.__name__, clf_C.__class__.__name__)
results2 = sub_dict(results, somekeys)
vs.evaluate(results2, accuracy, fscore)


# In[55]:


# print results
for i in range(3):
    for name in somekeys:
        print "classifier:", name,results[name][i]['acc_val']


# In[56]:


# print results
for i in range(3):
    for name in somekeys:
        print "classifier:", name, results[name][i]['f_val']


# In[60]:


# print results
for i in range(3):
    for name in somekeys:
        print "classifier:", name, float(results[name][i]['pred_time'])/results[name][i]['train_time']


# ----
# ## 提高效果
# 
# 在这最后一节中，您将从三个有监督的学习模型中选择 *最好的* 模型来使用学生数据。你将在整个训练集（`X_train`和`y_train`）上使用网格搜索优化至少调节一个参数以获得一个比没有调节之前更好的 F-score。

# ### 问题 3 - 选择最佳的模型
# 
# *基于你前面做的评价，用一到两段话向 *CharityML* 解释这三个模型中哪一个对于判断被调查者的年收入大于 \$50,000 是最合适的。*             
# **提示：**你的答案应该包括评价指标，预测/训练时间，以及该算法是否适合这里的数据。

# **回答：**
# 从准确率、F-Score分析，LogisticRegression 在1%，10%，100%时，得分均优于svc、RandomForestClassifier。本数据模型结果为二分类问题，并且计算量非常小，速度很快，存储资源低，因此其是适合本数据分析的。

# ### 问题 4 - 用通俗的话解释模型
# 
# *用一到两段话，向 *CharityML* 用外行也听得懂的话来解释最终模型是如何工作的。你需要解释所选模型的主要特点。例如，这个模型是怎样被训练的，它又是如何做出预测的。避免使用高级的数学或技术术语，不要使用公式或特定的算法名词。*

# **回答： ** 
# 逻辑回归适合分类结果只有两个值的情况，通过特定的方法将值限定在0-1之间，在模型的左侧为负、另一侧为正。

# ### 练习：模型调优
# 调节选择的模型的参数。使用网格搜索（GridSearchCV）来至少调整模型的重要参数（至少调整一个），这个参数至少需尝试3个不同的值。你要使用整个训练集来完成这个过程。在接下来的代码单元中，你需要实现以下功能：
# 
# - 导入[`sklearn.model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 和 [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - 初始化你选择的分类器，并将其存储在`clf`中。
#  - 设置`random_state` (如果有这个参数)。
# - 创建一个对于这个模型你希望调整参数的字典。
#  - 例如: parameters = {'parameter' : [list of values]}。
#  - **注意：** 如果你的学习器有 `max_features` 参数，请不要调节它！
# - 使用`make_scorer`来创建一个`fbeta_score`评分对象（设置$\beta = 0.5$）。
# - 在分类器clf上用'scorer'作为评价函数运行网格搜索，并将结果存储在grid_obj中。
# - 用训练集（X_train, y_train）训练grid search object,并将结果存储在`grid_fit`中。
# 
# **注意：** 取决于你选择的参数列表，下面实现的代码可能需要花一些时间运行！

# # 4.0.1 GridSearchCV + GaussianNB

# In[29]:


# still error
# GridSearchCV + GaussianNB
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.get_params()

# TODO：创建你希望调节的参数列表tions)
param_grid = {'priors':[0.875, 0.625]}

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer, cv=None)
print 'GridSearchCV'
t0 = time()
# print 'X_train_len:', len(X_train)
# print 'y_train_len:', len(y_train)

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
# grid_obj.fit(np.array(X_train[:100].reshape(100, 103)), np.array(y_train[:100].reshape(100, 103)))
# grid_obj.fit([X_train[:1000]] [y_train[:1000]])
print 'fit:', time() - t0

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)
print 'fit:',time() - t1

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''


# # 4.0.2 GridSearchCV + SGDClassifier 

# In[30]:


# GridSearchCV + SGDClassifier
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier

# TODO：创建你希望调节的参数列表
parameters = { "alpha" : np.logspace(-5,-3,5) }

param_grid = { "alpha" : np.logspace(-5,-3,5), 'loss':['hinge','log','modified_huber']}

n = 25000
# TODO：初始化分类器
# clf = SGDClassifier(loss ="log", average = 10,penalty = "l2",n_jobs = 3, n_iter = np.ceil(10**6 / n))
clf = SGDClassifier(random_state=35)

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer, cv=10, n_jobs=4)

t0 = time()
grid_obj.fit(X_train, y_train)
# TODO：用训练数据拟合网格搜索对象并找到最佳参数
print "fit time:", time() - t0

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train, y_train)).predict(X_val)
print "predict time:", time() - t1
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''
fit time: 18.0989999771
predict time: 0.101000070572
Unoptimized model
------
Accuracy score on validation data: 0.8383
F-score on validation data: 0.6823

Optimized Model
------
Final accuracy score on the validation data: 0.8380
Final F-score on the validation data: 0.6834
'''


# # 4.0.3 GridSearchCV + LogisticRegression

# In[31]:


'''
# GridSearchCV + LogisticRegression
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO：初始化分类器
clf = LogisticRegression(random_state=15)

# TODO：创建你希望调节的参数列表
# solver_options = ['saga', 'lbfgs']
class_weight_options = ['balanced']
penalty_options = ['l2']
C_options = [1,10,100]
solver_options = ['newton-cg','lbfgs','sag','saga']
param_grid = dict(solver = solver_options, class_weight = class_weight_options, C=C_options, penalty=penalty_options)

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer, cv=None)

grid_obj.fit(X_train, y_train)
# TODO：用训练数据拟合网格搜索对象并找到最佳参数

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''

'''
Unoptimized model
------
Accuracy score on validation data: 0.8536
F-score on validation data: 0.7182

Optimized Model
------
Final accuracy score on the validation data: 0.8097
Final F-score on the validation data: 0.6186
'''


# # 4.0.4 GridSearchCV + SVC

# In[32]:


# GridSearchCV + SVC
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer

# TODO：初始化分类器
clf = SVC(random_state=20)
print 'clf'
# TODO：创建你希望调节的参数列表
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001,0.00001]}

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer, cv=None)
print 'GridSearchCV'
t0 = time()
print 'X_train_len:', len(X_train)
print 'y_train_len:', len(y_train)

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
grid_obj.fit(X_train[:1000], y_train[:1000])
print 'fit:',time() - t0

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train[:1000], y_train[:1000])).predict(X_val)
best_predictions = best_clf.predict(X_val)
print 'predict:',time() - t1

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''
clf
GridSearchCV
X_train_len: 28941
y_train_len: 28941
fit: 7.23399996758
predict: 0.881000041962
Unoptimized model
------
Accuracy score on validation data: 0.7687
F-score on validation data: 0.2938

Optimized Model
------
Final accuracy score on the validation data: 0.8167
Final F-score on the validation data: 0.6310
'''


# # 4.0.5 GridSearchCV + KNN

# In[33]:


# GridSearchCV + KNN
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
# run time too long

from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer

# TODO：初始化分类器
clf = KNeighborsClassifier()
print 'clf'

# TODO：创建你希望调节的参数列表
params = {'n_neighbors': [2, 3, 5], 
          'weights': ['uniform', 'distance']}

k_range = range(1, 31)
weight_options = ['uniform', 'distance']
param_dist = dict(n_neighbors=k_range, weights=weight_options)

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
# grid_obj = GridSearchCV(clf, param_grid=params, scoring=scorer,cv=None)
grid_obj = RandomizedSearchCV(clf, param_dist, scoring=scorer,cv=10, n_iter=10, random_state=5)
print 'GridSearchCV'

print 'X_train_len:', len(X_train)
print 'y_train_len:', len(y_train)

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
t0 = time()
grid_obj.fit(X_train[:1000], y_train[:1000])
print 'fit time:', time() - t0
# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train, y_train)).predict(X_val)
print 'predict time:', time() - t1
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''
clf
GridSearchCV
X_train_len: 28941
y_train_len: 28941
fit time: 2.07100009918
fit time: 23.1059999466
Unoptimized model
------
Accuracy score on validation data: 0.8219
F-score on validation data: 0.6422

Optimized Model
------
Final accuracy score on the validation data: 0.8257
Final F-score on the validation data: 0.6521
'''


# # 4.0.6 GridSearchCV + DecisionTreeClassifier

# In[34]:


# GridSearchCV + DecisionTreeClassifier
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO：初始化分类器
clf = DecisionTreeClassifier(random_state=0)
print 'clf'

# TODO：创建你希望调节的参数列表
max_depth_options = list(range(1,22,5))
splitter_options = ['best', 'random']
max_features_options = ['sqrt', 'log2']

param_grid = dict(max_depth=max_depth_options, splitter=splitter_options, max_features=max_features_options)

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer,cv=None)
print 'GridSearchCV'

# TODO：用训练数据拟合网格搜索对象并找到最佳参数
t0 = time()
grid_obj.fit(X_train, y_train)
print 'fit time:', time() - t0
# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
t1 = time()
predictions = (clf.fit(X_train, y_train)).predict(X_val)
print 'fit time:', time() - t1
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
''' result
fit time: 5.5680000782
fit time: 0.483999967575
Unoptimized model
------
Accuracy score on validation data: 0.8214
F-score on validation data: 0.6401

Optimized Model
------
Final accuracy score on the validation data: 0.8443
Final F-score on the validation data: 0.6996
'''


# # 4.0.7 GridSearchCV + AdaBoostClassifier

# In[35]:


# GridSearchCV + AdaBoostClassifier
# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier

# TODO：初始化分类器
clf = AdaBoostClassifier()
print 'clf'

# TODO：创建你希望调节的参数列表
parameters_adaboost = {'n_estimators': [10, 20, 30, 40, 60, 100], 'algorithm' : ['SAMME', 'SAMME.R'], 'random_state': [40]}

# TODO：创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf, parameters_adaboost, scoring=scorer,cv=None)
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
print 'fit time:', time() - t1
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
'''
clf
GridSearchCV
fit time: 5.4509999752
fit time: 0.221999883652
Unoptimized model
------
Accuracy score on validation data: 0.8483
F-score on validation data: 0.6986

Optimized Model
------
Final accuracy score on the validation data: 0.8567
Final F-score on the validation data: 0.7260
'''


# # 4.0.8 GridSearchCV + RandomForestClassifier

# In[45]:


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
'''
clf
GridSearchCV
fit time: 19.1210000515
predict time: 0.0309998989105
Unoptimized model
------
Accuracy score on validation data: 0.8307
F-score on validation data: 0.6630

Optimized Model
------
Final accuracy score on the validation data: 0.8552
Final F-score on the validation data: 0.7219
'''


# ### 问题 5 - 最终模型评估
# 
# _你的最优模型在测试数据上的准确率和 F-score 是多少？这些分数比没有优化的模型好还是差？你优化的结果相比于你在**问题 1**中得到的天真预测器怎么样？_  
# **注意：**请在下面的表格中填写你的结果，然后在答案框中提供讨论。

# #### 结果:
#  
# | 评价指标         | 天真预测器           | 未优化的模型        | 优化的模型        |
# | :------------: | :-----------------: | :---------------: | :-------------: | 
# | 准确率          |  0.2478          |     0.8299         |     0.8527            |
# | F-score        |   0.2917          |     0.6615         |         0.7119        |

# **回答：**
# 未优化的模型、优化的模型准确率、F-score大幅度由于天真预测器，天真预测器的预测精度太低，用其获得的结果不可靠，且优化的模型比未优化的模型有所提高，符合期望。

# ----
# ## 特征的重要性
# 
# 在数据上（比如我们这里使用的人口普查的数据）使用监督学习算法的一个重要的任务是决定哪些特征能够提供最强的预测能力。专注于少量的有效特征和标签之间的关系，我们能够更加简单地理解这些现象，这在很多情况下都是十分有用的。在这个项目的情境下这表示我们希望选择一小部分特征，这些特征能够在预测被调查者是否年收入大于\$50,000这个问题上有很强的预测能力。
# 
# 选择一个有 `'feature_importance_'` 属性的scikit学习分类器（例如 AdaBoost，随机森林）。`'feature_importance_'` 属性是对特征的重要性排序的函数。在下一个代码单元中用这个分类器拟合训练集数据并使用这个属性来决定人口普查数据中最重要的5个特征。

# ### 问题 6 - 观察特征相关性
# 
# 当**探索数据**的时候，它显示在这个人口普查数据集中每一条记录我们有十三个可用的特征。             
# _在这十三个记录中，你认为哪五个特征对于预测是最重要的，选择每个特征的理由是什么？你会怎样对他们排序？_

# **回答：**
# - 特征1:age
# 年龄：处于就业年龄的收入大于为就业，且40岁左右的人正处于人生巅峰，此时收入更高的可能性更大；
# 
# - 特征2:education-num
# 受教育的年限越长，综合素质越高，因此收入更高的可能性越大；
# 
# - 特征3:occupation
# 金融、经济、高技能的职业收入更高；
# 
# - 特征4:hours-per-week
# 同等收入下，工作时间越长取得的收入越高；
# 
# - 特征5:capital-gain
# capital-gain越高，收入越多；
# 

# ### 练习 - 提取特征重要性
# 
# 选择一个`scikit-learn`中有`feature_importance_`属性的监督学习分类器，这个属性是一个在做预测的时候根据所选择的算法来对特征重要性进行排序的功能。
# 
# 在下面的代码单元中，你将要实现以下功能：
#  - 如果这个模型和你前面使用的三个模型不一样的话从sklearn中导入一个监督学习模型。
#  - 在整个训练集上训练一个监督学习模型。
#  - 使用模型中的 `'feature_importances_'`提取特征的重要性。

# In[37]:


# TODO：导入一个有'feature_importances_'的监督学习模型
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()

# TODO：在训练集上训练一个监督学习模型
model = clf.fit(X_train, y_train)

# TODO： 提取特征重要性
importances = model.feature_importances_
importances


# In[38]:


# Display the five most important features
# 获取列名
indices = np.argsort(importances)[::-1]
columns = X_train.columns.values[indices[:5]]
columns


# In[39]:


values = importances[indices][:5]
values


# In[40]:


# 绘图
vs.feature_plot(importances, X_train, y_train)


# ### 问题 7 - 提取特征重要性
# 观察上面创建的展示五个用于预测被调查者年收入是否大于\$50,000最相关的特征的可视化图像。
# 
# _这五个特征的权重加起来是否超过了0.5?_<br>
# _这五个特征和你在**问题 6**中讨论的特征比较怎么样？_<br>
# _如果说你的答案和这里的相近，那么这个可视化怎样佐证了你的想法？_<br>
# _如果你的选择不相近，那么为什么你觉得这些特征更加相关？_

# **回答：**
# 1. 权重和超过0.5，为0.56；
# 2. 大致符合，capital-loss出乎意料，居然权值是最高的；
# 3. 资本利得、年龄、受教育年龄、工作时间影响收入水平；
# 4. 不同职业的收入不一样，比如银行的收入大于厂里工人的收入。

# ### 特征选择
# 
# 如果我们只是用可用特征的一个子集的话模型表现会怎么样？通过使用更少的特征来训练，在评价指标的角度来看我们的期望是训练和预测的时间会更少。从上面的可视化来看，我们可以看到前五个最重要的特征贡献了数据中**所有**特征中超过一半的重要性。这提示我们可以尝试去**减小特征空间**，简化模型需要学习的信息。下面代码单元将使用你前面发现的优化模型，并**只使用五个最重要的特征**在相同的训练集上训练模型。

# In[61]:


column_import = X_train.columns.values[(np.argsort(importances)[::-1])[:5]]
column_import


# In[42]:


X_train[column_import].head(5)


# In[43]:


# 导入克隆模型的功能
from sklearn.base import clone

# 减小特征空间
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_val_reduced = X_val[X_val.columns.values[(np.argsort(importances)[::-1])[:5]]]

# 在前面的网格搜索的基础上训练一个“最好的”模型
clf_on_reduced = (clone(best_clf)).fit(X_train_reduced, y_train)

# 做一个新的预测
reduced_predictions = clf_on_reduced.predict(X_val_reduced)

# 对于每一个版本的数据汇报最终模型的分数
print "Final Model trained on full data\n------"
print "Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, reduced_predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, reduced_predictions, beta = 0.5))


# ### 问题 8 - 特征选择的影响
# 
# *最终模型在只是用五个特征的数据上和使用所有的特征数据上的 F-score 和准确率相比怎么样？*  
# *如果训练时间是一个要考虑的因素，你会考虑使用部分特征的数据作为你的训练集吗？*

# **回答：**
# 1. 基于五个特征的和所有特征相比，准确度、F-score出现小幅度的下降，我觉得是正常的，因为特征数越多，训练的模型准确率越高。
# 2. 不会，训练时间和特定的算法有关，并不能体现数据潜在的联系、类别。

# ### 问题 9 - 在测试集上测试你的模型
# 
# 终于到了测试的时候，记住，测试集只能用一次。
# 
# *使用你最有信心的模型，在测试集上测试，计算出准确率和 F-score。*
# *简述你选择这个模型的原因，并分析测试结果*

# In[44]:


#TODO test your model on testing data and report accuracy and F score
# X_train, X_test, y_train, y_test 
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]
y_pred = clf_on_reduced.predict(X_test_reduced)
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on validation data: {:.4f}".format(accuracy_score(y_test, y_pred))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_test, y_pred, beta = 0.5))


# > **注意：** 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)**把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。
