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

### 2.6 Word2Vec
Word2Vec算法，是一种简单而且计算时更加高效的词嵌入算法。
#### 2.6.1 Skip-Gram
Word2Vec中的Skip-Gram模型，所做的是在语料库中选定某个词（Context），随后在该词的正负10个词距内取一些目标词（Target）与之配对，构造一个用Context预测输出为Target的监督学习问题，训练一个如下图结构的网络：

![word2vec_1.jpg](https://i.imgur.com/GTacllg.jpg)

该网络仅有一个softmax单元，输出context下target出现的条件概率：

![word2vec_2.jpg](https://i.imgur.com/wKIvh8H.jpg)

θ<sub>t</sub>：是一个与输出Target有关的参数，即某个词和标签相符的概率是多少。

e<sub>c</sub>=E \* O<sub>c</sub>

损失函数：

![word2vec_3.png](https://i.imgur.com/V33g7Qi.png)

y是只有一个1其他都是0的one-hot向量

#### 2.6.2 Softmax分类器
实际上使用这个算法会遇到一些问题，首要的问题就是计算速度。改进方法是使用分级Softmax分类器(Hierarchical Softmax Classifier)，采用霍夫曼树（Huffman Tree）来代替隐藏层到输出Softmax层的映射。

![word2vec_4.png](https://i.imgur.com/gyxnA3I.png)

上图中，树上内部的每一个节点都可以是一个二分类器，比如逻辑回归分类器，所以不需要再为单次分类，对词汇表中所有的10,000个词求和了。实际上用这样的分类树，计算成本与词汇表大小的对数成正比，而不是词汇表大小的线性函数。

实际上，softmax分类器不会使用一棵完美平衡的分类树或者说一棵左边和右边分支的词数相同的对称树(上图左侧)；

通常分级的softmax分类器会被构造成常用词在顶部，然而不常用的词像durian会在树的更深处(上图右侧)。

#### 2.6.3 怎么对上下文c进行采样？
一种选择是你可以就对语料库均匀且随机地采样，但是像the、of、a、and、to诸如此类是出现得相当频繁的。实际上词p(c)的分布并不是单纯的在训练集语料库上均匀且随机的采样得到的，而是采用了不同的分级来平衡更常见的词和不那么常见的词。

#### 2.6.4 Skip-Gram CBOW
CBOW，它获得中间词两边的的上下文，然后用周围的词去预测中间的词。

总结下：CBOW是从原始语句推测目标字词；而Skip-Gram正好相反，是从目标字词推测出原始语句。CBOW对小型数据库比较合适，而Skip-Gram在大型语料中表现更好。 （下图左边为CBOW，右边为Skip-Gram）

![word2vec_5.png](https://i.imgur.com/5gyUkqN.jpg)

### 2.7 负采样(Negative Sampling)