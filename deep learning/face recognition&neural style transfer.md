### 4.0 Face verification vs. face recognition
![face_recognition_1.png](https://i.imgur.com/ljqsqvk.png)

人脸验证一般指一个一对一问题，只需要验证输入的人脸图像等信息是否与某个已有的身份信息对应；

人脸识别需要验证输入的人脸图像是否与多个已有的信息中的某一个匹配，是一个更为复杂的一对多问题。
### 4.1 One-shot leanring
在一次学习问题中，只能通过一个样本进行学习，以能够认出同一个人。大多数人脸识别系统都需要解决这个问题，因为在数据库中每个雇员或者组员可能都只有一张照片。

在识别过程中，如果这两张图片的差异值小于某个阈值τ，它是一个超参数，那么这时就能预测这两张图片是同一个人，如果差异值大于τ，就能预测这是不同的两个人

#### Similarity函数
d(img1, img2) = degree of difference between images
if d(img1, img2) <= τ then true
else false

### 4.3 Siamese 网络(Siamese network)
Siamese 网络实现了Similarity函数

对于两个不同的输入，运行相同的卷积神经网络，然后比较它们，这一般叫做Siamese网络架构。

![siamese_1.png](https://i.imgur.com/HbhaYEv.png)

神经网络的参数定义了一个编码函数f(x<sup>(i)</sup>)，如果给定输入图像x<sup>(i)</sup>，这个网络会输出x<sup>(i)</sup>的128维的编码。训练得到学习参数，使得如果两个图片x<sup>(i)</sup>和x<sup>(j)</sup>是同一个人，那么得到的两个编码的距离就小。相反，如果x<sup>(i)</sup>和x<sup>(j)</sup>是不同的人，那么它们之间的编码距离大一点。

### 4.4 Triplet损失
Triplet损失函数的定义基于三张图片–两张同一人的不同人脸图像和一张其他人的人脸图像，它们的特征向量分别用符号A（Anchor）、P（Positive）、N（Negative）表示。

度量距离函数：d(A,P)=||f(A)-f(P)||<sup>2</sup> <= d(A,N)=||f(A)-f(N)||<sup>2</sup>

为避免d(A,P)=d(A,N)=0时，上诉函式成立，改进：d(A,P)=||f(A)-f(P)||<sup>2</sup> - d(A,N)=||f(A)-f(N)||<sup>2</sup> + α <= 0，α是margin。

损失函数：L(A,P,N) = max(||f(A)-f(P)||<sup>2</sup> - ||f(A)-f(N)||<sup>2</sup> + α,0)

只要这个损失函数小于等于0，网络不会关心它负值有多大。

![Triplet_1.png](https://i.imgur.com/OKEo0JU.png)

这是一个三元组定义的损失，整个网络的代价函数应该是训练集中这些单个三元组损失的总和。然后用梯度下降最小化定义的代价函数。

![Triplet_2.png](https://i.imgur.com/wNFOu5x.png)

如果随机的选择A,P,N那么训练将很容易满足，但这样训练得到的网络意义不大。所以需要尽可能选择难训练的三元组A,P,N。

具体而言，你想要所有的三元组都满足这个条件d(A,P)+α<=d(A,N)，难训练的三元组就是A,P,N选择使得d(A,P)很接近d(A,N)，即d(A,P)约等于d(A,N)，这样学习算法会竭尽全力使右边这个式子变大d(A,N)，或者使左边这个式子d(A,P)变小，这样左右两边至少有一个的间隔。并且选择这样的三元组还可以增加你的学习算法的计算效率。

### 4.5 面部验证与二分类(Face verification and binary classification)
这一节介绍其他学习人脸识别参数的方法，如何将人脸识别当成一个二分类问题。

选取一对Siamese网络，使其同时计算这些嵌入，比如说128维的嵌入（编号1），或者更高维，然后将其输入到逻辑回归单元，然后进行预测，如果是相同的人，那么输出是1，若是不同的人，输出是0。这就把人脸识别问题转换为一个二分类问题，训练这种系统时可以替换Triplet loss的方法。

![Triplet_3.png](https://i.imgur.com/M2g3CeD.png)

逻辑回归单元的处理：

![Triplet_4.png](https://i.imgur.com/E3kxwcV.png)

f(x<sup>(i)</sup>)<sub>k</sub>代表图片x<sup>(i)</sup>的编码，下标k代表选择这个向量中的第k个元素，|f(x<sup>(i)</sup>)<sub>k</sub>-f(x<sup>(j)</sup>)<sub>k</sub>|对这两个编码取元素差的绝对值，逻辑回归可以增加参数w<sub>i</sub>和b。

绿色标记是其他不同的形式来计算绝对值。

因为不需要存储原始图像，如果有一个很大的员工数据库，不需要为每个员工每次都计算这些编码；当一个新员工走近时，可以使用上方的卷积网络来计算这些编码，然后使用它，和预先计算好的编码进行比较，然后输出预测值yhat。这个预先计算的思想，可以节省大量的计算，这个预训练的工作可以用在Siamese网路结构中，将人脸识别当作一个二分类问题，也可以用在学习和使用Triplet loss函数上。

总结一下，把人脸验证当作一个监督学习，创建一个只有成对图片的训练集，不是三个一组，而是成对的图片，目标标签是1表示一对图片是一个人，目标标签是0表示图片中是不同的人。利用不同的成对图片，使用反向传播算法去训练神经网络，训练Siamese神经网络。

### 4.6 什么是神经风格转换？(what is neural stype transfer)
略
### 4.7 什么是深度卷积网络？(what are deep convNets learning)
浅层的隐藏单元通常学习到的是边缘、颜色等简单特征，越往深层，隐藏单元学习到的特征也越来越复杂。

### 4.8 代价函数(cost function)
内容图像C，风格图片S，生成一个新图片G，为了实现神经风格迁移，定义一个关于的代价函数G用来评判某个生成图像的好坏，使用梯度下降法去最小化J(G)，以便于生成这个图像。

J(G)=αJ<sub>content</sub>(C,G) + βJ<sub>style</sub>(S,G)

梯度下降：

![style_transfer_1.png](https://i.imgur.com/MTQxwkS.png)

### 4.9 内容代价函数(content cost function)
![content_cost_function_1.png](https://i.imgur.com/cPLpYrt.png)

令α<sup>[l][C]</sup>和α<sup>[l][G]</sup>代表这两个图片C和G的l层激活函数值，如果这两个激活值相似，那么就意味着两个图片的内容相似。

定义：J<sub>content</sub>(C,G)=0.5 * ||α<sup>[l][C]</sup> - α<sup>[l][G]</sup>||<sup>2</sup>为两个激活值不同或者相似的程度，取l层的隐含单元的激活值，按元素相减，内容图片的激活值与生成图片相比较，然后取平方，也可以在前面加上归一化或者不加。

### 4.10 风格代价函数(style cost function)
暂略