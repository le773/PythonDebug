### 1.0 neural gpu
#### 1.1 cgru definition
cgru definition(similar to GRU replacing linear by convolution):</br>
s<sub>t+1</sub>=g\*s<sub>t</sub> + (1-g)\*c where:</br>
g = sigmoid(conv(s<sub>t</sub>, K<sub>g</sub>))</br>
c = tanh(conv(s<sub>t</sub> \* r, K<sub>α</sub>))</br>
r = sigmoid(conv(s<sub>t</sub>, K<sub>r</sub>))

![ngpu_1.png](https://i.imgur.com/ltKw0Le.png)

#### 1.2 Computational power
- Small number of parameters(3 kernels:K<sub>g</sub>, K<sub>α</sub>,K<sub>r</sub>)
- Can simulate computation of cellular automata(可模拟元胞自动机的计算)
- With memory of size n can do n local operations / step
- E.g.., can do long multiplication in O(n) steps

----------

### 2.0 sequence to sequence
![seq2seq_7.png](https://i.imgur.com/tGjd1d3.jpg)

一个Seq2Seq模型中可分成编码器（Encoder）和译码器（Decoder）两部分，它们通常是两个不同的神经网络。

这种Enconder-Decoder的结构，也可以应用在图像标注（Image Caption）上；

![seq2seq_8.png](https://i.imgur.com/dK5G1tX.jpg)

上图中，将图像输入了一个作为编码器的AlexNet结构的CNN中，最后的softmax换成一个RNN作为译码器，训练网络输出图像的标注结果。

在机器翻译用到的seq2seq模型中，译码器所做的工作与前面讲过的语言模型的采样过程类似，只不过在机器翻译中，用编码器的输出代替语言模型中的0作为译码器中第一个时间步的输入，如下图：

![seq2seq_9.png](https://i.imgur.com/19wzKpD.jpg)

采用大量的数据训练好一个机器翻译系统后，对于一个相同的句子，由于译码器进行的是随机采样过程，输出的可能会是多种或好或坏的结果；所以对训练好的机器翻译系统，还需要加入一些算法，使其总是输出最好的翻译结果。比如贪心搜索、集束搜索等。

----------

### 2.1 sequence to sequence问题
sequence to sequence是一类机器学习问题。

![seq2seq_4.png](https://i.imgur.com/yqnmClp.png)

从统计模型的角度看，sequence to sequence问题，就是已知在序列F出现的情况下，找到另一个序列出现的条件概率最大的Ehat,其中θ是模型的参数，E和F是两个随机出现的序列。

sequence to sequence问题解决步骤
1. 找到合适的模型，用来表示P(E|F;θ)
2. 训练模型参数
3. 找到最可能的序列

### 2.2 Encoder-Decoder模式
1.word2vec:接受一个或N个词，返回一个或N个词。</br>
原理：a.将输入的词编码(Encoder)为一个固定长度的向量；</br>
b.通过神经网络预测和这个向量相关的另一个向量；</br>
c.将得到的词向量解码(Decoder)为对词典的概率分布。

2.autoencoder：输入一张图片，返回生成的新图片</br>
a.将输入的图片，通过卷积神经网络，编码(Encoder)为一个固定长度的向量(图片的特征向量)</br>
b.通过转置卷神经网络(逆变换)，将特征向量解码(Decoder)为一张新的图片。

----------

### 2.3 Encoder-Decoder模式的seq2seq
1.将输入的序列，通过某种结构，编码为一个固定长度的向量
2.通过某种逆向结构，将特征向量解码为一个新的序列

### 2.4 seq2seq
1. 通过RNN网络(Encoder)，将输入序列映射为一个固定长度的向量
2. 在利用另一个RNN网络(Decoder)，将向量映射为一个序列。

![seq2seq_5.png](https://i.imgur.com/AzZSv3W.png)

![seq2seq_6.png](https://i.imgur.com/o7aZOBU.png)

m<sub>t</sub><sup>(f)</sup>表示输入序列在时间t的特征向量(比如词向量)。</br>
m<sub>t</sub><sup>e</sup>表示输出序列在时间t的特征向量。</br>
RNN(.)表示循环神经网络。</br>
h表示循环神经网络的隐状态。</br>
p表示输出序列在时间t的概率分布。


### 2.5 seq2seq的输出
1. 随机采样，在每一步输出，根据概率分布，随机采样选择一个输出；
2. 贪婪策略，在每一步输出，选择概率最大的作为输出；
3. 集束搜索(Beam Search)，在每一步预测是留下一定数量的分支，进行下一步。

![seq2seq_1.png](https://i.imgur.com/aCnJiaq.png)

### 3.1 集束搜索(Beam Search)
第一步，设定束长(Beam Width)B，它代表了译码器中每个时间步的预选单词数量。将第一个时间步中预测出的概率最大的3个词作为首词的预选词，同时保存它们的概率值大小p(y<sup>(1)</sup>|x);

![seq2seq_11.png](https://i.imgur.com/81kfQxG.jpg)

如果第一步得到的三个预选词分别为“in”、“jane”和“September”，如下图所示，则第二步中，分别将三个预选词作为第一个时间步的预测结果y<sup>(1)</sup>输入第二个时间步，得到预测结果yhat<sup>(2)</sup>也就是条件概率值p(yhat<sup>(2)</sup>|x,y<sup>(1)</sup>):

![seq2seq_12.png](https://i.imgur.com/332Zb1y.jpg)

根据条件概率公式，有：

p(y<sup>(1)</sup>, yhat<sup>(2)</sup>|x) = p(y<sup>(1)</sup>|x) \* p(yhat<sup>(2)</sup>|x, y<sup>(1)</sup>)

分别以三个首词预选词作为y<sup>(1)</sup>进行计算，将得到3\*10000个p(y<sup>(1)</sup>，yhat<sup>(2)</sup>|x)。之后还是取其中概率值最大的B=3个，作为对应首词条件下的第二个词的预选词。后面的过程以此类推，最后将得到一个最优的翻译结果。

### 3.2 改进集束搜索(Beam Search)
总的来说，集束搜索算法所做的工作就是找出符合以下公式的结果：

![beam_search_1.png](https://i.imgur.com/kpow129.png)

然而概率值都是小于1的值，多个概率值相乘后的结果的将会是一个极小的浮点值，累积到最后的效果不明显且在一般的计算机上达不到这样的计算精度。改进的方法，是取上式的log值并进行标准化：

![beam_search_2.png](https://i.imgur.com/POIMWNe.png)

如果参照原来的目标函数(上上图)，如果有一个很长的句子，那么这个句子的概率会很低，因为乘了很多项小于1的数字来估计句子的概率。所以如果乘起来很多小于1的数字，那么就会得到一个更小的概率值，所以这个目标函数有一个缺点，它可能不自然地倾向于简短的翻译结果，它更偏向短的输出，因为短句子的概率是由更少数量的小于1的数字乘积得到的，所以这个乘积不会那么小。顺便说一下，这里也有同样的问题，概率的log值通常小于等于1，实际上在的这个范围内，所以加起来的项越多，得到的结果越负，所以对这个算法另一个改变也可以使它表现的更好，也就是不再最大化这个目标函数了，可以把它归一化，通过除以翻译结果的单词数量。这样就是取每个单词的概率对数值的平均了，这样很明显地减少了对输出长的结果的惩罚。

![beam_search_3.png](https://i.imgur.com/MhD3aUN.png)

如果α等于1，就相当于完全用长度来归一化，如果α等于0，就相当于完全没有归一化。α实际上是试探性的，它并没有理论验证。但是大家都发现效果很好。

![beam_search_4.png](https://i.imgur.com/mwi93W0.png)

### 3.3 关于束长B的取值
关于束长B的取值，较大的B值意味着同时考虑了更多的可能，最后的结果也可能会更好，但会带来巨大的计算成本；较小的B值减轻了计算成本的同时，也可能会使最后的结果变得糟糕。通常情况下，B值取一个10以下地值较为合适。还是要根据实际的应用场景，适当地选取。要注意的是，当B=1时，这种算法就和贪心搜索算法没什么两样了。

### 3.4 集束搜索(Beam Search)的误差分析
集束搜索是一种启发式（Heuristic）搜索算法，它的输出结果不是总为最优的。

![beam_search_5.png](https://i.imgur.com/WBhcuJ7.jpg)

上图中，人工翻译和算法存在较大的差别，要找到错误的根源，首先将翻译没有差别的一部分“Jane visits Africa”分别作为译码器中其三个时间步的输入，得到第四个时间步的输出为“in”的概率p(y<sup>*</sup>|x)和“last”的概率p(y^∣x)，比较它们的大小并分析：

- 如果p(y<sup>*</sup>|x) > p(y^∣x)，那么说明是集束搜索时出现错误，没有选择到概率最大的词；
- 如果p(y<sup>*</sup>|x) < p(y^∣x)，那么说明是RNN模型的表现不够好，预测的第四个词为“in”的概率小于“last”。

### 3.5 BLEU(Bilingual Evaluation Understudy)指标
最原始的BLEU算法很简单：统计机器翻译结果中的每个单词在参考翻译中出现的次数作为分子，机器翻译结果的总词数作为分母。然而这样得到结果很容易出现错误。

![bleu_1.png](https://i.imgur.com/24fGoGQ.jpg)

上面的方法是一个词一个词进行统计，这种以一个单词为单位的集合统称为uni-gram（一元组）。以uni-gram统计得到的精度p1体现了翻译的充分性，也就是逐字逐句地翻译能力。

两个单词为单位的集合则称为bi-gram（二元组）:

![bleu_2.png](https://i.imgur.com/b1Oyuv3.jpg)

以二元组的统计有：

![bleu_3.png](https://i.imgur.com/ZR0S5qJ.png)

根据上表，计算机器翻译的精度，有4/6。

推广到n元组(n-gram):

![bleu_4.png](https://i.imgur.com/2qiyGyU.png)

以n-gram统计得到的pn体现了翻译的流畅度。将uni-gram下的p1到n-gram下的pn组合起来，对这N个值进行几何加权平均得到：

![bleu_5.png](https://i.imgur.com/4D0ZjcT.png)

此外，注意到采用n-gram时，机器翻译的结果在比参考翻译短的情况下，很容易得到较大的精度值。改进的方法是设置一个最佳匹配长度（Best Match Length），机器翻译的结果未达到该最佳匹配长度时，则需要接受简短惩罚（Brevity Penalty，BP）：

![bleu_6.png](https://i.imgur.com/VniiWSg.png)

最后，得到BLEU指标为：

![bleu_7.png](https://i.imgur.com/Vkgh7sR.png)

### 4.0 seq2seq的改进
##### 4.1.1 采用逆向编码
![seq2seq_2.png](https://i.imgur.com/Am9nL2E.png)

正向编码中，输出序列某个词，与输入序列中相关词的距离约等于输入序列的长度。考虑到RNN网络存在梯度消失的问题，如果依赖距离过长，会导致无法有效更新参数。

逆序编码后，可以对于序列中的一些词降低依赖距离。训练过程中，短距离依赖可以更快的收敛，并同事可以帮助长距离依赖得到更好的结果。

##### 4.1.2 双向RNN
![seq2seq_3.png](https://i.imgur.com/m9pWZ6E.png)

### seq2seq的问题
1.长词依赖(尽管有些技巧可以降低影响)。
2.不同长度的序列，使用相同长度的特征向量。

### 5.0 Attention
1.计算输入序列中每一个词的隐状态。
2.在解码时，读取对应的隐状态。

![seq2seq_attention_1.png](https://i.imgur.com/fSQzEob.png)

![seq2seq_attention_2.png](https://i.imgur.com/NcvvV86.png)

![seq2seq_attention_3.png](https://i.imgur.com/KdL29Hp.png)

h<sub>t</sub>隐状态。
### 5.1 Attention Score的计算
![seq2seq_attention_score_1.png](https://i.imgur.com/gPy01Ag.png)


### 6.0 注意力模型的直观理解
注意力模型中，网络的示例结构如下所示：

![AttentionModel_1.png](https://i.imgur.com/4Yh0rUs.jpg)

底层是一个双向循环神经网络，需要处理的序列作为它的输入。该网络中每一个时间步的激活a<sup><t'></sup>中，都包含前向传播产生的和反向传播产生的激活：

![AttentionModel_2.png](https://i.imgur.com/KytBpEX.png)

顶层是一个多对多结构的循环神经网络，第t个时间以该网络中前一个时间步的激活s<sup><t-1></sup>、输出y<sup><t-1></sup>以及底层的BRNN中多个时间步的激活c作为输入。对第t个时间步的输入c有：

![AttentionModel_3.png](https://i.imgur.com/NXYnNjw.png)

其中的参数a<sup><t,t'></sup>意味着顶层RNN中，第t个时间步输出的y<sup><t></sup>中，把多少注意力放在了底层BRNN的第t'时间步的激活a<sup><t'></sup>上。它总有：

![AttentionModel_4.png](https://i.imgur.com/Cfov13e.png)

为确保参数a<sup><t,t'></sup>满足上式，常用softmax单元来计算顶层RNN的第t个时间步对底层BRNN第t'个时间步的激活的注意力：

![AttentionModel_5.png](https://i.imgur.com/DW0zwZ7.png)

其中e<sup><t,t'></sup>由顶层RNN的激活s<sup><t-1></sup>和底层BRNN的激活a<sup><t'></sup>一起输入一个隐藏层中得到的，因为e<sup><t,t'></sup>也就是a<sup><t,t'></sup>的值明显与s<sup><t></sup>、a<sup><t'></sup>有关，由于s<sup><t></sup>此时还是未知量，则取上一层的激活s<sup><t-1></sup>。在无法获知s<sup><t-1></sup>、a<sup><t'></sup>与e<sup><t,t'></sup>之间的关系下，那就用一个神经网络来进行学习，如下图：

![AttentionModel_6.png](https://i.imgur.com/vWQve8z.jpg)

要注意的是，该模型的运算成本将要达到N*N。