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
Word2Vec算法，只有一个隐层的全连接神经网络, 用来预测给定单词的关联度大的单词，是一种简单而且计算时更加高效的词嵌入算法。

![word2vec_7.png](https://i.imgur.com/GVChKDC.jpg)

1. 在输入层，一个词被转化为One-Hot向量。
2. 然后在第一个隐层，输入的是一个w \* x+b(x就是输入的词向量，w,b是参数)，做一个线性模型，注意已这里只是简单的映射，并没有非线性激活函数，当然一个神经元可以是线性的，这时就相当于一个线性回归函数。
3. 第三层可以简单看成一个分类器，用的是Softmax回归，最后输出的是每个词对应的概率
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

![word2vec_6](https://i.imgur.com/oPjLhwn.png)

### 2.7 负采样(Negative Sampling)
使用霍夫曼树来代替传统的神经网络，可以提高模型训练的效率。但是如果我们的训练样本里的中心词w是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了。能不能不用搞这么复杂的一颗霍夫曼树，将模型变的更加简单呢？
#### 2.7.1 负采样(Negative Sampling)步骤
![word2vec_8.png](https://i.imgur.com/MpD1Fxj.jpg)

生成一个正样本，先抽取一个上下文词，在一定词距内比如说正负10个词距内选一个目标词，标签设为1，这就是生成这个表的第一行；</br>
然后为了生成一个负样本，使用相同的上下文词，再在字典中随机选一个词，标签设置为0；

#### 那么如何选取K？
Mikolov等人推荐小数据集的话，K从5到20比较好。如果数据集很大，K就选的小一点。对于更大的数据集K就等于2到5，数据集越小就越大。

#### 负采样(Negative Sampling)
接下来构造监督学习问题。原网络中的Softmax变成多个Sigmoid单元，给定输入的c,t对的条件下，y=1(正样本)的概率，即：

![word2vec_10.png](https://i.imgur.com/Qo9t29j.png)

这个模型基于逻辑回归模型，其中的θ<sub>t</sub>、e<sub>c</sub>分别代表Target及Context的词向量。
通过这种方法将之前的一个复杂的多分类问题变成了多个简单的二分类问题，而降低计算成本。

模型中采用以下公式来计算选择某个词作为负样本的概率：

![word2vec_9.png](https://i.imgur.com/tQXAmKg.png)

其中f(w<sub>i</sub>)代表语料库中单词w<sub>i</sub>出现的频率。

总结：在softmax分类器能够学到词向量，但是计算成本很高。通过负采样将其转化为一系列二分类问题，因为不用计算所有单词的出现的概率，所以可以非常有效的学习词向量。

[刘建平：word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)

[Word2Vec介绍: 为什么使用负采样（negtive sample）？](https://zhuanlan.zhihu.com/p/29488930)

### 2.8 GloVe 词向量(GloVe Word Vectors)
代价函数：

![glove_1.png](https://i.imgur.com/Zzg5UA5.png)

X<sub>i,j</sub>:表示整个语料库中单词i和单词j彼此接近的频率，也就是它们共同出现在一个窗口中的次数。

θ<sub>i</sub>,e<sub>j</sub>分别是单词i和单词j的词向量，b<sub>i</sub>,b<sub>j</sub>是两个偏差项，f()是一个用以防止X<sub>i,j</sub>=0时，log(X<sub>i,j</sub>)无解的权重函数，词汇表的大小为N。

### 2.9 情感分类(Sentiment Classification)
![sentiment_classification_1.png](https://i.imgur.com/owVpkPh.png)

#### 简单的方法
![sentiment_classification_2.png](https://i.imgur.com/8bSLhyi.png)

对语句中每个词的特征向量求和或平均，然后得到一个表示300维的特征向量，然后把使用softmax计算。

这个算法有一个问题就是没考虑词序，尤其是这样一个负面的评价，"Completely lacking in good taste, good service, and good ambiance."，但是good这个词出现了很多次，有3个good，如果算法跟这个一样，**忽略词序**，仅仅把所有单词的词嵌入加起来或者平均下来，最后的特征向量会有很多good的表示，分类器很可能认为这是一个好的评论，尽管事实上这是一个差评，只有一星的评价。

#### 使用RNN做情感分类
![sentiment_classification_3.png](https://i.imgur.com/UzFFMRS.png)

用每一个one-hot向量乘以词嵌入矩阵E，得到词嵌入表达e，然后把它们送进RNN里。RNN的工作就是在最后一步计算一个特征表示，用来预测yhat。

训练一个这样的算法，最后会得到一个很合适的情感分类的算法。由于词嵌入是在一个更大的数据集里训练的，这样效果会更好，更好的泛化一些没有见过的新的单词。

### 2.10 词嵌入除偏(Debiasign word embeddings)
对于性别歧视这种情况来说，“男性（Man）”对“程序员（Computer Programmer）”将得到类似“女性（Woman）”对“家务料理人（Homemaker）”的性别偏见结果。

1.中和本身与性别无关词汇
对某词向量，将50维空间分为两部分：与性别相关的方向g和与g正交的其他49个维度。

![debiasing_word_embedding_1.jpg](https://i.imgur.com/EdIskWH.jpg)

除偏的步骤，是将要除偏的词向量，左图中的e<sub>receptionist</sub>，在向量g方向上的值置为0，变成右图所示的e<sup>debiased</sup><sub>receptionist</sub>，公式如下：

![debiasing_word_embedding_2.jpg](https://i.imgur.com/rZAcEhj.jpg)

2.均衡本身与性别有关词汇

![debiasing_word_embedding_3.jpg](https://i.imgur.com/s5FiKtt.jpg)

均衡过程的核心思想是确保一对词（actor和actress）到g⊥的距离相等的同时，也确保了它们到除偏后的某个词（babysit）的距离相等，如上右图。

对需要除偏的一对词w1、w2，选定与它们相关的某个未中和偏见的单词B之后，均衡偏见的过程如下公式：

![debiasing_word_embedding_4.jpg](https://i.imgur.com/wdEoNwe.jpg)
