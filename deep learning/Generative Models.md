## Generative Models
Given training data, generate new samples from same distribution

![GenerativeModels_1.png](https://i.imgur.com/aASjiLv.png)

### Why Generative Models?
- Realistic samples for artwork, super-resolution, colorization, etc.
- Generative models of time-series data can be used for simulation and planning (reinforcement learning applications!)
- Training generative models can also enable inference of latent representations that can be useful as general features

### Taxonomy of Generative Models
![GenerativeModels_2.png](https://i.imgur.com/M913l5z.png)


### Fully visible belief network
Explicit density model

Use chain rule to decompose likelihood of an image x into product of 1-d distributions:

![fvbn_1.png](https://i.imgur.com/ygkPpxY.png)

Then maximize likelihood of training data

PixelRNN和PixelCNN都属于全可见信念网络，要做的是对一个密度分布显示建模。我们有图像数据x，想对该图像的概率分布或者似然p(x)建模，我们使用链式法则将这一似然分解为一维分布的乘积，我们有每个像素xi的条件概率，其条件是给定所有下标小于i的像素（x1到xi-1），这时图像中所有像素的概率或联合概率就是所有这些像素点似然的乘积。一旦定义好这些似然，为了训练好这一模型，我们只需要在该定义下最大化我们的训练数据的似然。

右边的像素值概率分布，也就是给定所有在xi之前的像素值条件的条件概率p(xi)，下面我们会用神经网络表达这一概率分布的复杂函数。

### PixelRNN
Generate image pixels starting from corner</br>
Dependency on previous pixels modeled using an RNN (LSTM)

![PixelRNN_1.png](https://i.imgur.com/zqhBHt6.png)

从左上角一个一个生成像素，生成顺序为箭头所指顺序，每一个对之前像素的依赖关系都通过RNN来建模。</br>
缺点：顺序生成的，速度会很慢。

### PixelCNN
![PixelCNN_2.png](https://i.imgur.com/q0PfW70.png)

Still generate image pixels starting from corner</br>
Dependency on previous pixels now modeled using a CNN over context region</br>
Training: maximize likelihood of training images

![PixelCNN_1.png](https://i.imgur.com/ewBAiya.png)

从图像拐角处生成整个图像，区别在于现在使用CNN来对所有依赖关系建模。

现在对环境区域（图示那个指定像素点的附近灰色区域）上使用CNN，取待生成像素点周围的像素，把他们传递给CNN用来生成下一个像素值，每一个像素位置都有一个神经网络输出，该输出将会是像素的softmax损失值，我们通过最大化训练样本图像的似然来训练模型，在训练的时候取一张训练图像来执行生成过程，每个像素位置都有正确的标注值，即训练图片在该位置的像素值，该值也是我们希望模型输出的值。

1. Training is faster than PixelRNN (can parallelize convolutions since context region values known from training images)
2. Generation must still proceed sequentially

### PixelRNN PixelCNN
PixelRNN和PixelCNN能显式地计算似然p(x)，是一种可优化的显式密度模型，该方法给出了一个很好的评估度量，可以通过计算的数据的似然来度量出你的生成样本有多好。

----------

## Autoencoders变分自编码器
变分自编码器与自动编码器的无监督模型相关。</br>
我们不通过自动编码器来生成数据，它是一种利用无标签数据来学习低维特征的无监督学习。</br>

![Encoder_1.png](https://i.imgur.com/Ab7fyXP.png)

输入数据x和特征z，接下来我们有一个编码器进行映射，来实现从输入数据x到特征z的映射。编码器可以有多种形式，常用的是神经网络。最先提出的是非线性层的线性组合，又有了深层的全连接网络，又出现了CNN，我们取得输入数据x然后将其映射到某些特征z，我们再将z映射到比x更小的维度上，由此可以实现降维。</br>
对z进行降维是为了表示x中最重要的特征。


如何学习这样的特征表示？

![Encoder_2.png](https://i.imgur.com/zjwEawj.png)

自动编码器将该模型训练成一个能够用来重构原始数据的模型，我们用编码器将输入数据映射到低维的特征z（也就是编码器网络的输出），同时我们想获得基于这些数据得到的特征，然后用第二个网络也就是解码器输出一些跟x有相同维度并和x相似的东西，也就是重构原始数据。

对于解码器，我们一般使用和编码器相同类型的网络（通常与编码器对称）。

![Encoder_3.png](https://i.imgur.com/GJAWjoH.png)

整个流程：取得输入数据，把它传给编码器网络（比如一个四层的卷积网络），获取输入数据的特征，把特征传给解码器（四层的解卷积网络），在解码器末端获得重构的数据。

为了能够重构输入数据，我们使用L2损失函数，让输入数据中的像素与重构数据中的像素相同。

训练好网络后，我们去掉解码器。 

![Encoder_4.png](https://i.imgur.com/Q9feTCO.png)

使用训练好的编码器实现特征映射，通过编码器得到输入数据的特征，编码器顶部有一个分类器，如果是分类问题我们可以用它来输出一个类标签，在这里使用了外部标签和标准的损失函数如softmax。

我们可以用很多无标签数据来学习到很多普适特征，可以用学习到的特征来初始化一个监督学习问题，因为在监督学习的时候可能只有很少的有标签训练数据，少量的数据很难训练模型，可能会出现过拟合等其他一些问题，通过使用上面得到的特征可以很好的初始化网络。

自动编码器重构数据，学习数据特征，初始化一个监督模型的能力。这些学习到的特征具有能捕捉训练数据中蕴含的变化因素的能力。我们获得了一个含有训练数据中变化因子的隐变量z。

### Variational Autoencoders
Probabilistic spin on autoencoders - will let us sample from the model to generate data!

![VariationalAutoencoders_1.png](https://i.imgur.com/1a2FSak.png)

通过向自编码器加入随机因子获得的一种模型，我们能从该模型中采样从而生成新的数据。
我们有训练数据x（i的范围从1到N），数据x是从某种不可观测的隐式表征z中生成的。例如我们想要生成微笑的人脸，z代表的就是眉毛的位置，嘴角上扬的弧度。
生成过程：从z的先验分布中采样，对于每种属性，我们都假设一个它应该是一个怎样的分布。高斯分布就是一个对z中每个元素的一种自然的先验假设，同时我们会通过从在给定z的条件下，x的条件概率分布p(x|z)中采样。先对z采样，也就是对每个隐变量采样，接下来对图像x采样。

对于上述采样过程，真实的参数是Θ*，我们有关于先验假设和条件概率分布的参数，我们的目的在于获得一个生成式模型，从而利用他来生成新的数据，真实参数中的这些参数是我们想要估计并得出的。

### How should we represent this model?如何表述上述模型
Choose prior p(z) to be simple, e.g. Gaussian. Reasonable for latent attributes, e.g. pose, how much smile.

对上述过程建模，选一个简单的关于z的先验分布，例如高斯分布

Conditional p(x|z) is complex (generates image) => represent with neural network

对于给定z的x的条件概率分布p(x|z)很复杂，所以我们选择用神经网络来对p(x|z)进行建模。

### How to train the model?
Remember strategy for training generative models from FVBNs. Learn model parameters to maximize likelihood of training data.

![VariationalAutoencoders_2.png](https://i.imgur.com/B5WHIGC.png)

之后调用解码器网络，选取隐式表征，并将其解码为它表示的图像。训练一个生成模型，学习到一个对于这些参数的估计。
如何训练模型？最大化训练数据的似然函数来寻找模型参数， 在已经给定隐变量z的情况下，我们需要写出x的分布p并对所有可能的z值取期望，因为z值是连续的所以表达式是一个积分。
直接求导来求最大化的似然，过程会很不好解。

### Variational Autoencoders: Intractability
Data likelihood:

![VariationalAutoencoders_3.png](https://i.imgur.com/3wFpKwW.png)

Posterior density also intractable:

![VariationalAutoencoders_4.png](https://i.imgur.com/MWwQa16.png)

数据似然项p(x)，第一项是z的分布p(z)，这里将它简单地设定为高斯分布，p(x|z)指定一个神经网络解码器，这样一来任意给定一个z我们就能获得p(x|z)也就是神经网络的输出。但是计算每一个z对应的p(x|z)很困难，所以无法计算该积分。
数据的似然函数是难解的，导致了模型的其他项，后验密度分布也就是p(z|x)也是难解的。

Solution: In addition to decoder network modeling p<sub>θ</sub>(x|z), define additional encoder network q<sub>ɸ</sub>(z|x) that approximates p<sub>θ</sub>(z|x)

我们无法直接优化似然。解决方法是，在使用神经网络解码器来定义一个对p(x|z)建模神经网络的同时，额外定义一个编码器q(z|x)，将输入x编码为z，从而得到似然p(z|x)。也就是说我们定义该网络来估计出p(z|x)，这个后验密度分布项仍然是难解的，

Will see that this allows us to derive a lower bound on the data likelihood that is tractable, which we can optimize

我们用该附加网络来估计该后验分布，这将使我们得到一个数据似然的下界，该下界易解也能优化。

![VariationalAutoencoders_5.png](https://i.imgur.com/Do1557w.png)

Since we’re modeling probabilistic generation of data, encoder and decoder networks are probabilistic

在变分自编码器中我们想得到一个生成数据的概率模型，将输入数据x送入编码器得到一些特征z，然后通过解码器网络把z映射到图像x。我们这里有编码器网络和解码器网络，将一切参数随机化。参数是Φ的编码器网络q(z|x)输出一个均值和一个对角协方差矩阵；解码器网络输入z，输出均值和关于x的对角协方差矩阵。为了得到给定x下的z和给定z下的x，我们会从这些分布（p和q）中采样，现在我们的编码器和解码器网络所给出的分别是z和x的条件概率分布，并从这些分布中采样从而获得值。
编码器网路也是一种识别或推断网络，因为是给定x下对隐式表征z的推断；解码器网络执行生成过程，所以也叫生成网络。

Now equipped with our encoder and decoder networks, let’s work out the (log) data likelihood:

![VariationalAutoencoders_6.png](https://i.imgur.com/sM5KW4l.png)

如何得到下界：第一项是对所有采样的z取期望，z是x经过编码器网络采样得到，对z采样然后再求所有z对应的p(x|z)。让p(x|z)变大，就是最大限度地重构数据。第二项是让KL的散度变小，让我们的近似后验分布和先验分布变得相似，意味着我们想让隐变量z遵循我们期望的分布类型。

![VariationalAutoencoders_7.png](https://i.imgur.com/g4ayF6N.png)

得到了下界后，现在把所有东西整合到一起，过一遍如何训练自动编码器。
公式是我们要优化及最大化的下界，前向传播按如上流程处理，对输入数据x，让小批量的数据传递经过编码器网络的到q(z|x)，通过q(z|x)来计算KL项，然后根据给定x的z分布对z进行采样，由此获得了隐变量的样本，这些样本可以根据x推断获得；然后把z传递给第二个解码器网络，通过解码器网络x在给定z的条件下的两个参数，均值和协方差，最终可以在给定z的条件下从这个分布中采样得到x。
训练时需要获得该分布，损失项是给定z条件下对训练像素值取对数，损失函数要做的是最大化被重构的原始输入数据的似然；对于每一个小批量的输入我们都计算这一个前向传播过程，取得所有我们需要的项，他们都是可微分的，接下来把他们全部反向传播回去并获得梯度，不断更新我们的参数，包括生成器和解码器网络的参数Θ和Φ从而最大化训练数据的似然。

....
