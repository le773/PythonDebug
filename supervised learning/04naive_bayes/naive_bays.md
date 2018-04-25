### 01 贝叶斯
缺陷：在多个单词组成的意义明显不同的词语中，分类不太好

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

#### 01.05 优缺点
优点: 在数据较少的情况下仍然有效，可以处理多类别问题。
缺点: 对于输入数据的准备方式较为敏感。
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

参考:
[feature_extraction](http://scikit-learn.org/stable/modules/feature_extraction.html "feature_extraction")
[使用scikit-learn工具计算文本TF-IDF值](http://blog.csdn.net/eastmount/article/details/50323063 "使用scikit-learn工具计算文本TF-IDF值")

#### 03.02 词频

![词频](http://img.blog.csdn.net/20160808160728646)

#### 03.03 逆文档率

![逆文档率](http://img.blog.csdn.net/20160808160752037)

#### 03.04 tf-idf

![tf-idf](http://img.blog.csdn.net/20160808160817878)
