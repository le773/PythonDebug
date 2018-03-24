## 1.0 Density Estimation
### 1.1 Problem Motivation 异常检测
![problem_motivation](https://i.imgur.com/Xpkewy4.png)

### 1.2 Gaussian Distribution 高斯分布
![Gaussian_distribution](https://i.imgur.com/W72S4Y5.png)

### 1.3 Algotithm
![Anomaly_detection_algorithm_1](https://i.imgur.com/iSIF265.png)

## 2.0 Building an Anomaly Detection System
### 2.1 Developing and Evaluating an Anomaly Detection System

![the_importance_of_real_number_evaluation_1](https://i.imgur.com/WniAm7I.png)
为了更快地，开发出一个异常检测系统，那么最好能找到某种，评价异常检测系统的方法。

- 交叉验证集和测试集所用的数据应当独立，切勿混用。

###### 算法评估
![algorithm_evaluation_1](https://i.imgur.com/HgjlJGh.png)

### 2.2 Anomaly Detection vs. Supervised Learning
![Anomaly_detection_vs_Supervised_learning_1](https://i.imgur.com/R58mjCu.png)

### 2.3 Choosing What Features to Use
![Non_gaussian_to_gaussian_1](https://i.imgur.com/kNeLUlY.png)
1. 取对数log
2. 求根

#### 2.3.2 Error analysis for anomaly detectioin
![error_analysis_for_anomaly_detection_1](https://i.imgur.com/F6DepP7.png)
误差分析过程
寻找飞机引擎中不寻常的问题，然后建立一些新特征变量；通过这些新特征变量，从正常样本中区别异常。

#### 2.3.3 Monitoring computers in a data center
Choose features that might take on unusually large or small values in the event of an anomaly

选择在异常情况下，选择出现异常出现不非常大或不小值的特征向量.

## 3.0 Multivariate Gaussian Distribution(Optional)
#### 3.1 Multivariate Gaussian Distribution 多元高斯分布
###### 异常检测的缺陷
![motivating_monitoring_machines_1](https://i.imgur.com/FlM62Dg.png)


#### 3.1.3 多元高斯分布

![多元高斯分布](http://img.blog.csdn.net/20150621144912308)

- μ 相当于每个正态分布的对称轴，是一个一维向量
- Σ是协方差矩阵(n*n)

![Multivariate_gaussian_examples_1](https://i.imgur.com/7FfTMqJ.png)

#### 3.2 Anomaly Detection using the Multivariate Gaussian Distribution
![Multivariate_gaussian_distribution_1](https://i.imgur.com/eAdmaVF.png)

![Anomaly_detection_with_the_multivariate_gaussian_1](https://i.imgur.com/VfiJGtr.png)

![Relationship_to_original_model_1](https://i.imgur.com/Sw7I9UN.png)

###### 高斯模型vs原始模型
![original_vs_multivariate_gaussian_1](https://i.imgur.com/9TTk7EX.png)

需要捕获特征变量之间的相关性，一般手动增加额外的特征变量来捕获特定的不正常的值的组合，但是在训练集很大或者说m很大n不太大的情况下，高斯是值得考虑的。

多元高斯模型可以省去建立特征值组合来捕捉异常，而手动建立额外特征变量所花费的时间。

协方差矩阵Σ不可逆时：
1. 确保训练集合m比特征数量n大很多
2. 检查冗余特征

## 4.0 Predicting Movie Ratings
### 4.1.1 Problem Formulation 问题公式化
### 4.1.2 Content Based Recommendations
![problem_formulation_1](https://i.imgur.com/790XWIS.png)

![Optimization_Objective_2](https://i.imgur.com/VqWWDIy.png)

![Optimization_Objective_3](https://i.imgur.com/vUPKN8q.png)

## 5.0 Collaborative Filtering
### 5.1 Collaborative Filtering
![Optimization_Objective_4](https://i.imgur.com/SojfCyL.png)

![Collaborative_filter](https://i.imgur.com/AdGywZY.png)

协同过滤算法指的是:当你执行这个算法时,通过一大堆用户,得到的数据,这些用户实际上在高效地,进行了协同合作,来得到每个人,对电影的评分值,只要用户对某几部电影进行评分,每个用户就都在帮助算法,更好的学习出特征。这样通过自己,对几部电影评分之后,我就能帮助系统更好的学习到特征,这些特征可以被系统运用,为其他人做出更准确的电影预测;协同的另一层意思是说:每位用户都在为了大家的利益,学习出更好的特征,这就是协同过滤。 

### 5.2 Collaborative Filtering Algorithm
![Collaborative_filter_optimization_objective_1](https://i.imgur.com/m0hXUp7.png)

![Collaborative_filter_algorithm_1](https://i.imgur.com/q1m6JvQ.png)

## 6.0 Low Rank Matrix Factorization
### 6.1 Vectorization:Low Rank Matrix Factorization
![Collaborative_filter_low_rank_matrix_1](https://i.imgur.com/y18p9uJ.png)

### 6.2 Implementational Detail:Mean Normalization
