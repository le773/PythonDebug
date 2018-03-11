### 贝叶斯
缺陷：在多个单词组成的意义明显不同的词语中，分类不太好

### 02
#### CountVectorizer + TfidfTransformer
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
#### TfidfVectorizer
`there is also another class called TfidfVectorizer that **combines** all the options of **CountVectorizer** and **TfidfTransformer** in a single model`

TfidfVectorizer是CountVectorizer TfidfTransformer 的结合体。

```
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
```

### 03 TF-IDF基础知识
`Tf–idf term weighting`权重计算：
![Tf–idf term weighting](http://scikit-learn.org/stable/_images/math/40f34fb794a1d3561d64bc55e344634b1451a21f.png)

idf
![idf](http://scikit-learn.org/stable/_images/math/9b51d66bf06313c9ed7e2014ad2dae110e764d7b.png)
also
![idf](http://scikit-learn.org/stable/_images/math/d210fb6831f799e768f25c722773fe9912c1f7e3.png)
where ![nd](http://scikit-learn.org/stable/_images/math/7f022743140b3c69e2a3bb796a82bb989ff558af.png) is the total number of documents, and ![df](http://scikit-learn.org/stable/_images/math/a5f393d00e7621eca4b3334f87f15bd31752caa4.png) is the number of documents that contain term ![t](http://scikit-learn.org/stable/_images/math/5ec053cf70dc1c98cc297322250569eda193e7a4.png). The resulting tf-idf vectors are then normalized by the Euclidean norm:
![Vnorm](http://scikit-learn.org/stable/_images/math/1fa6fb7a6ac6f7a11b410e2ae6f61a2a52283292.png)
This was originally a term weighting scheme developed for information retrieval (as a ranking function for search engines results) that has also found good use in document classification and clustering.

nd:代表文章数量；
df(d,t):某词t在文章d中出现的次数；
参考:
[feature_extraction](http://scikit-learn.org/stable/modules/feature_extraction.html "feature_extraction")
[使用scikit-learn工具计算文本TF-IDF值](http://blog.csdn.net/eastmount/article/details/50323063 "使用scikit-learn工具计算文本TF-IDF值")

#### 示例

词频

![词频](http://img.blog.csdn.net/20160808160728646)

逆文档率

![逆文档率](http://img.blog.csdn.net/20160808160752037)

tf-idf

![tf-idf](http://img.blog.csdn.net/20160808160817878)
