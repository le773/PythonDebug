### 02
```
# 将文本转为词频矩阵
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
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