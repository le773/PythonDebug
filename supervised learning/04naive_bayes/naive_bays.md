### 01 贝叶斯
##### 贝叶斯公式
![navies_bayes_1.jpg](https://i.imgur.com/ksOUOwJ.jpg)

上图理解为：

![navies_bayes_2.jpg](https://i.imgur.com/rHCkIvI.jpg)

通俗实例：

![navies_bayes_3.jpg](https://i.imgur.com/QVFLZoF.jpg)

#### 01.02 为什么朴素贝叶斯被称为"朴素"?

这个等式成立的条件需要特征之间相互独立。这一假设使得朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率。

这也就是为什么朴素贝叶斯分类有朴素一词的来源，朴素贝叶斯算法是假设各个特征之间相互独立，那么这个等式就成立了！

#### 01.03 但是为什么需要假设特征之间相互独立呢？

我们这么想，假如没有这个假设，那么我们对右边这些概率的估计其实是不可做的，这么说，我们这个例子有4个特征，其中帅包括{帅，不帅}，性格包括{不好，好，爆好}，身高包括{高，矮，中}，上进包括{不上进，上进}，那么四个特征的联合概率分布总共是4维空间，总个数为2*3*3*2=36个。

24个，计算机扫描统计还可以，但是现实生活中，往往有**非常多的特征，每一个特征的取值也是非常之多，那么通过统计来估计后面概率的值，变得几乎不可做**，这也是为什么需要假设特征之间独立的原因。

#### 01.04 工作原理
首先，统计去重后的词条在各个类别中的数量，各类别总数量；

其次，计算概率：P(词条|类别)*P(类别)
```
提取所有文档中的词条并进行去重
获取文档的所有类别
计算每个类别中的文档数目
对每篇训练文档: 
    对每个类别: 
        如果词条出现在文档中-->增加该词条的计数值（for循环或者矩阵相加）
        增加所有词条的计数值（此类别下词条总数）
对每个类别: 
    对每个词条: 
        将该词条的数目除以总词条数目得到的条件概率（P(词条|类别)）
返回该文档属于每个类别的条件概率（P(类别|文档的所有词条)）
```
上述步骤，并未计算P(词条)的概率，因为需要比较P(类别|文档的所有词条)的大小，也就是比较P(词条|类别)*P(类别)的值。

#### 01.05 拉普拉斯校验
由于训练量不足，某个类别下某个特征划分没有出现时，会令分类器质量大大降低。为了解决这个问题，引入Laplace校准（这就引出了我们的拉普拉斯平滑）。

它的思想非常简单，就是对没类别下所有划分的计数加1，这样如果训练样本集数量充分大时，并不会对结果产生影响，并且解决了上述频率为0的尴尬局面。

![laplace_1.png](https://i.imgur.com/3dmf5Aq.png)

![laplace_2.png](https://i.imgur.com/h7Nw8Hd.png)

其中a<sub>jl</sub>代表第j个特征的第l个选择，S<sub>j</sub>代表第j个特征的个数，K代表种类数

λ=1，加入拉普拉斯平滑后，避免了出现概率为0的情况。
#### 01.06 特征离散型
当特征是离散型，则直接统计。

#### 01.07 特征连续型
#### 01.07.01 利用高斯分布
假设连续变量服从某种概率分布，然后使用训练数据估计分布的参数，高斯分布通常被用来表示连续属性的类条件概率分布。

高斯分布有两个参数，均值μ和方差σ<sup>2</sup>，对每个类y<sub>i</sub>，属性X<sub>i</sub>的类条件概率等于：

![gaussian_1.png](https://i.imgur.com/aPBo2EK.png)

参数μ<sub>ij</sub>可以用类y<sub>j</sub>的所有训练记录关于X<sub>i</sub>的样本均值来估计，同理，σ<sub>ij</sub><sup>2</sup>可以用这些训练记录样本方差来估计。

#### 01.07.02 核密度估计
核密度估计(Kernel density estimation)，是一种用于估计概率密度函数的非参数方法，x<sub>1</sub>,x<sub>2</sub>,...,x<sub>n</sub>为独立同分布F的n个样本点，设其概率密度函数为f，核密度估计为以下：

![kernel_density_estimation_1.png](https://i.imgur.com/Qh7RBbK.png)

K()为核函数，各种核函数如下：

![kernel_density_estimation_2.png](https://i.imgur.com/ZPhKUwO.png)

#### 01.08 朴素贝叶斯优缺点
优点:

1.在数据较少的情况下仍然有效，可以处理多类别问题。

缺陷：

1.在多个单词组成的意义明显不同的词语中，分类不太好

适用数据类型: 标称型数据。

### 02 CountVectorizer + TfidfTransformer
#### 02.01 CountVectorizer + TfidfTransformer 代码实现
```
# 将文本转为词频矩阵
#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词
vectorizer=CountVectorizer()在i类文本下的词频
#该类会统计每个词语的tf-idf权值
transformer=TfidfTransformer() 
# corpus 切词后的文本
# fit_transform 将文本转为词频矩阵
w_matrix = vectorizer.fit_transform(corpus)
#fit + transform
#获取词袋模型中的所有词语  
word=vectorizer.get_feature_names()
#计算tf-idf
tfidf=transformer.fit_transform(w_matrix)
#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
weight=tfidf.toarray()
```

#### 02.02 TfidfVectorizer
`there is also another class called TfidfVectorizer that combines all the options of CountVectorizer and TfidfTransformer in a single model`

`TfidfVectorizer`是`CountVectorizer` `TfidfTransformer`的结合体。

```
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
```

### 03 TF-IDF基础知识
#### 03.01 `Tf–idf term weighting`权重计算：
![Tf–idf term weighting](http://scikit-learn.org/stable/_images/math/40f34fb794a1d3561d64bc55e344634b1451a21f.png)

idf
![idf](http://scikit-learn.org/stable/_images/math/9b51d66bf06313c9ed7e2014ad2dae110e764d7b.png)
also
![idf](http://scikit-learn.org/stable/_images/math/d210fb6831f799e768f25c722773fe9912c1f7e3.png)
where ![nd](http://scikit-learn.org/stable/_images/math/7f022743140b3c69e2a3bb796a82bb989ff558af.png) is the total number of documents, and ![df](http://scikit-learn.org/stable/_images/math/a5f393d00e7621eca4b3334f87f15bd31752caa4.png) is the number of documents that contain term ![t](http://scikit-learn.org/stable/_images/math/5ec053cf70dc1c98cc297322250569eda193e7a4.png). The resulting tf-idf vectors are then normalized by the Euclidean norm:
![Vnorm](http://scikit-learn.org/stable/_images/math/1fa6fb7a6ac6f7a11b410e2ae6f61a2a52283292.png)
This was originally a term weighting scheme developed for information retrieval (as a ranking function for search engines results) that has also found good use in document classification and clustering.

`nd`:代表文章数量；
`df(d,t)`:某词t在文章d中出现的次数；

#### 03.02 词频

![词频](http://img.blog.csdn.net/20160808160728646)

#### 03.03 逆文档率

![逆文档率](http://img.blog.csdn.net/20160808160752037)

#### 03.04 tf-idf

![tf-idf](http://img.blog.csdn.net/20160808160817878)

----------

### 04 Q&A
#### 04.1 为什么特征独立型的模型遇到高度相关特征效果会不好？
1. 若冗余特征过多，会造成特征数目过多，从而分析特征，训练模型所需要的时间就会越长；
2. 冗余特征会使得并没有增加输入信息的前提下增加模型判别的置信度，这显然是不合理的。

以贝叶斯为例，贝叶斯的推导公式：

![navie_bays.png](https://i.imgur.com/3m2E2ya.png)

其中，先验p(y)不会改变，也不会产生影响。所以只需要看右面部分，分母分为两项，其中一项包含分子,不妨记右边部分为ϕ(x,y)。

以一个具体例子来说明，假设有这样一个二分类问题，特征是年龄和收入，分类标签是是否可以贷款，不妨设：

先验概率：
```
P(贷) = 0.7，P(不贷) = 0.3
```
似然函数：
```
P(年龄<50|贷) = 0.7，P(年龄>50|贷) = 0.3
P(收入>1w|贷) = 0.6，P(收入<1w|贷) = 0.3
P(年龄<50|不贷) = 0.8，P(年龄>50|不贷) = 0.2
P(收入>1w|不贷) = 0.1，P(收入<1w|贷) = 0.9
````
那么计算：

![navie_bays_2.png](https://i.imgur.com/15rLPQV.png)

其中分母两项的比值为 0.7x0.6/0.2x0.1=21

当存在冗余特征的时候，不妨假设特征存在重复，此时计算：

![navie_bays_3.png](https://i.imgur.com/8rIEH9M.png)

其中分母两项的比值为441

可以看到后者算出来的比值也变成了前者的平方，模型分类的置信度增加了很多，这实际上是不合理的，因为我们并没有增加数据上的额外信息在里面，只是单纯的特征重复，就使得分类置信度提高了很高，模型在分类的时候也会由于重复特征的影响造成分类效果的下降。

参考:

[为什么特征独立型的模型遇到高度相关特征效果会不好？](https://blog.csdn.net/shijing_0214/article/details/75864342)

[feature_extraction](http://scikit-learn.org/stable/modules/feature_extraction.html "feature_extraction")

[使用scikit-learn工具计算文本TF-IDF值](http://blog.csdn.net/eastmount/article/details/50323063 "使用scikit-learn工具计算文本TF-IDF值")

[核密度估计(Kernel density estimation)](https://blog.csdn.net/yuanxing14/article/details/41948485)


[为什么朴素贝叶斯分类中引入的拉普拉斯修正是正确的？](https://www.zhihu.com/question/41043365)