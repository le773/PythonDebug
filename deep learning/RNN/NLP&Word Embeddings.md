### 2.1 词汇表征(word representation)
#### one-hot表示方法

![word_representation.png](https://i.imgur.com/XXunNwn.png)

one-hot表示的缺点就是它把每个词孤立起来，这样使得算法对相关词的泛化能力不强。
#### 特征表示方法
![word_representation_2.png](https://i.imgur.com/5268M5F.png)

假设每个词有300个不同的特征，这样的话就有了这一列数字，King、Queen有性别特征，所以值接近+1、-1。这种高维特征的表示能够比one-hot更好的表示不同的单词。

![word_representation_3.png](https://i.imgur.com/9tHfEZD.png)

如果能够学习到一个300维的特征向量，或者说300维的词嵌入，通常把这300维的数据嵌入到一个二维空间里，进行可视化。常用的可视化算法是t-SNE算法。

t-SNE算法所做的就是把这些300维的数据用一种非线性的方式映射到2维平面上，可以得知t-SNE中这种映射很复杂而且很非线性。

![t-SNE_mapping_1.jpg](https://i.imgur.com/hua5jtj.jpg)

上图中，各词单词根据它们的语义及相关程度，分别汇聚在了一起。

词嵌入(word embedding)将单词word映射到另外一个空间，其中这个映射具有injective和structure-preserving的特点。
1. injective（就是单射函数，每个Y只有唯一的X对应，反之亦然）
2. structure-preserving (结构保存，比如在X所属的空间上X1 < X2,那么映射后在Y所属空间上同理 Y1 < Y2)。

[word embedding的含义？](https://www.zhihu.com/question/32275069)
### 2.2 使用词嵌入(Using Word Embeddings)
<font face="Consolas">
Transfer learning and word embedding
1. learn word embedding from large text corpus(1-100B words) (or downloaded pre-trained embedding online)
2. Transfer embedding to new task with smaller training set.(say, 100k words)
3. Optional:Continue to finetune the word embedding with new data.
</font>

第一步，先从大量的文本集中学习词嵌入。一个非常大的文本集，或者可以下载网上预训练好的词嵌入模型，网上你可以找到不少，词嵌入模型并且都有许可。

第二步，可以用这些词嵌入模型把它迁移到新的只有少量标注训练集的任务中。

第三步，当在新的任务上训练模型时，在命名实体识别任务上，只有少量的标记数据集上，可以自己选择要不要继续微调，用新的数据调整词嵌入。实际中，只有这个第二步中有很大的数据集才这样做，如果标记的数据集不是很大，通常不会在微调词嵌入上费力气。

当任务的训练集相对较小时，词嵌入的作用最明显，所以它广泛用于NLP领域。

迁移学习：从某一任务A迁移到某个任务B，只有A中有大量数据，而B中数据少时，迁移的过程才有用。

### 2.3 词嵌入的特性(Properties of Word Embeddings)
![word_embedding_1.png](https://i.imgur.com/Lj8wUPI.png)

找到单词w最大化上述公式。

度量词向量的相似度(余弦相似度)：

![word_representation_4.png](https://i.imgur.com/jTMWxpX.png)

![cosine_similarity_1.png](https://i.imgur.com/EkuWabt.png)

u,v<sup>T</sup>是两个向量的点积，|u|<sub>2</sub>是向量u的范数，并且θ是向量u,v之间的角度。这种相似性取决于角度在向量u和v之间。如果向量u和v非常相似，它们的余弦相似性将接近1; 如果它们不相似，则余弦相似性将取较小的值。 

![cosine_similarity_2.png](https://i.imgur.com/P4aWjoh.png)

### 2.4 嵌入矩阵(Embedding Matrix)
![embedding_matrix_1.png](https://i.imgur.com/XqKlQfa.png)

将字典中位置为i的词以one-hot形式表示为o<sub>i</sub>，嵌入矩阵用E表示，词嵌入后生成的词向量用e<sub>i</sub>表示。

E * o<sub>i</sub> = e<sub>i</sub>

### 2.5 学习词嵌入(learning word embeddings)
![learn_word_embedding_1.png](https://i.imgur.com/jVtu0dh.png)

经过神经网络以后再通过softmax层，这个softmax分类器会在10,000个可能的输出中预测结尾这个单词。通过预测目标词，就能得到嵌入矩阵，通过上述的数学公式就能得到词嵌入后的词向量。

![learn_word_embedding_2.png](https://i.imgur.com/WczuPhs.png)

研究者发现，如果想建立一个语言模型，用目标词的前几个单词作为上下文是常见做法。

如果目标是学习词嵌入，那么可以用这些其他类型的上下文，它们也能得到很好的词嵌入。
