# 图像分割
图像分割需要将前景和背景分隔开，即对背景打上掩码。
1. 普通的图像分割就是按照图像的颜色纹理将图片的不同区域划分开来；
2. 语义分割(Semantic Segmentation)是将不同的单元划分出来，比如将人、树、草地分割开来；实例（instances）分割则更进一步，将人这个单元的不同个体给分割开来，从而知道左边的人和右边的人不是同一个实例个体。

## 语义分割(Semantic Segmentation)
![Semantic Segmentation_1.png](https://i.imgur.com/m2Ep1XE.png)
1. 为像素打上相应的标签(Label every pixel!)
2. 不关心具体实例(Don’t differentiate instances (cows))

对每个像素，在卷积神经网络中进行分类：

![Semantic Segmentation_2.png](https://i.imgur.com/shBOrh5.png)

Run “fully convolutional” network to get all pixels at once

![Semantic Segmentation_4.png](https://i.imgur.com/6rl6i0U.png)

Smaller output due to pooling、strides etc..

Multi-Scale

![Semantic Segmentation_5.png](https://i.imgur.com/EFaAEd0.png)

Refinement

![Semantic Segmentation_6.png](https://i.imgur.com/OUgXekI.png)

### Semantic Segmentation Idea: Sliding Window
![Sliding Window_1.png](https://i.imgur.com/9TCyYzv.png)

Problem: Very inefficient! Not reusing shared features between overlapping patches
### Semantic Segmentation Idea: Fully Convolutional
Design a network as a bunch of convolutional layers to make predictions for pixels all at once!

![Fully Convolutional_1.png](https://i.imgur.com/fUpI1XM.png)

Problem: convolutions at original image resolution will be very expensive ...(使用原始图像会使运算效率很低，所以我们会在图像内使用下采样，对特征做上采样。)

![Fully Convolutional_2.png](https://i.imgur.com/csYSwGO.png)


Corresponding pairs of downsampling and upsampling layers

缩小图像（或称为下采样（subsampled）或降采样（downsampled））的主要目的有两个：1、使得图像符合显示区域的大小；2、生成对应图像的缩略图。

池化等

放大图像（或称为上采样（upsampling）或图像插值（interpolating））的主要目的是放大原图像,从而可以显示在更高分辨率的显示设备上。


#### In-Network upsampling: “Max Unpooling”
![max_unpooling_1.png](https://i.imgur.com/YpX9463.png)
#### In-Network upsampling: “Nearest Neighbor”
![upsample2.png](https://i.imgur.com/M68s8LF.png)

对图像的缩放操作并不能带来更多关于该图像的信息, 因此图像的质量将不可避免地受到影响。然而，确实有一些缩放方法能够增加图像的信息，从而使得缩放后的图像质量超过原图质量的。

 下采样原理：对于一副图像I尺寸为M*N，对起进行s倍下采样，即得到（M/s）*（N/s）尺寸的分辨率图像，当然，s应该是M和N的公约数才可以，如果考虑是矩阵形式的图像，就是把原始图像s*s窗口内的图像编程一个像素，这个像素点的值就是窗口内所有像素的均值。

Pk = Σ Ii / s2

上采样原理：图像放大几乎都是采用内插值方法，即在原有图像像素的基础上在像素点之间采用合适的插值算法插入新的元素。

插值算法还包括了传统插值，基于边缘图像的插值，还有基于区域的图像插值。

Instance Segmentation: Hypercolumns

![Semantic Segmentation_7.png](https://i.imgur.com/Ftp2IF2.png)

### Learnable Upsampling: Transpose Convolution
[卷积的反向传播及实现](https://github.com/le773/PythonDebug/blob/master/deep%20learning/convolutional%20neural%20network.md)

### Classification + Localization
![Classification + Localization](https://i.imgur.com/4Nl72CN.png)

输入一幅图像，Alex网络，现在有两个全连接层，一个是图像目标分类的预测，还有一个是向量（高度，宽度，x，y坐标）代表了目标在图像中的位置。训练网络时有两组损失，目标分类用softmax损失函数，边界输出损失用L2损失，评定预测边界和真实边界之间的差距。

其它：Classification + Localization的一种应用是Human Pose Estimation(人体姿势预测)

### Object Detection
在对象识别中，不确定图像中有几个对象（之前的分类问题和定位图片中只有一个目标）。所以要输出多个目标的分类预测和边界预测。

#### Object Detection as Regression?
Object Detection as Regression是很棘手的，因为Each image needs a different number of outputs!

![ObjectDetection_1.png](https://i.imgur.com/vCdsnMq.png)

#### Object Detection as Classification: Sliding Window
![ObjectDetection_2.png](https://i.imgur.com/ZxYULuv.png)

Problem: Need to apply CNN to huge number of locations, scales, and aspect ratios, very computationally expensive!

但是存在问题：对象数目不确定，并且可能以任何大小比例位置出现在图像中，如果输入所有可能的分割区域，那么计算量很大， 所以实际中一般不采用。

#### Region Proposals / Selective Search
- Find “blobby” image regions that are likely to contain objects
- Relatively fast to run; e.g. Selective Search gives 2000 region
proposals in a few seconds on CPU

![ObjectDetection_3.png](https://i.imgur.com/1y9u8DB.png)


#### R-CNN
![R-CNN_3.png](https://i.imgur.com/qe7CUUg.png)

给定输入，运行区域选择网络。这些区域的尺寸大小可能不一致，在卷积网络中因为全连接层的特性我们希望所有输入尺寸一致，所以要对这些区域调整为固定尺寸，使之与下游网络输入相匹配。

调整尺寸后输入卷积神经网络，然后使用支持向量机基于样本做分类。网络不止对区域做分类，还会对边界做修正和补偿。

#### R-CNN:Problems

- Ad hoc training objectives
• Fine-tune network with softmax classifier (log loss)
• Train post-hoc linear SVMs (hinge loss)
• Train post-hoc bounding-box regressions (least squares)
- Training is slow (84h), takes a lot of disk space
- Inference (detection) is slow
• 47s / image with VGG16 [Simonyan & Zisserman. ICLR15]
• Fixed by SPP-net [He et al. ECCV14]

### Fast R-CNN
![Fast R-CNN_1.png](https://i.imgur.com/fNPQ3j1.png)

不再按感兴趣区域分，而是将图片输入卷积层，得到高分辨率特征映射，针对备选区域切分图像像素，基于备选区域投影到卷积特征映射，从卷积特征映射提取属于备选区域的卷积块，而不是直接截取备选区域。

从卷积层映射提取到的图像块的尺寸进行调整（感兴趣区域池化），然后输入到全连接层预测分类结果和边界包围盒的线性补偿。

#### Fast R-CNN:RoI Pooling
![Fast R-CNN_2.png](https://i.imgur.com/dpybkIK.png)

#### Fast R-CNN:反向求导
![Fast R-CNN_3.png](https://i.imgur.com/WEk2S2g.png)

#### Faster R-CNN
![Faster R-CNN_1.png](https://i.imgur.com/4j08J7H.png)

Insert Region Proposal Network (RPN) to predict proposals from features Jointly train with 4 losses:
1. RPN classify object / not object
2. RPN regress box coordinates
3. Final classification score (object classes)
4. Final box coordinates

Fast R-CNN耗时在计算备选区域上了，之前使用固定方法计算备选区域，现在让网络自身去学习预测。</br>
在卷积层中运行整个图像，获取特征映射来表示整个高清晰度图像。分离备选区域网络工作在卷积特征上层，在网络中预测备选区域。从这些备选区域及特征映射中取出块，作为输入传至下一层。</br>
多任务损失及多任务训练网络，区域选择网络做两件事：
1. 对每个备选区域，他们是否是待识别物体；
2. 对包围盒进行校正。网络最后还需要做这两件事。


#### Detection without Proposals: YOLO / SSD
![SSD_1.png](https://i.imgur.com/boLLBn8.png)

YOLO:You Only Look Once</br>
SSD:Single-Shot MultiBox Detector

输入图像分成网格，例子中采用7\*7网格，在每一个单元中有一系列边界框，对每个单元和每个基本边界框预测几种目标物体：
1. 预测边界框偏移，
2. 从而预测出边界框与目标物体位置的偏差，
3. 预测目标对应的分类。

输出：7*7*（5*B+C）：B个基本边界框，每个边界框对应5个值，分别对应边界框的差值和我们的置信度，C对应C类目标类别的分数。

### Instance Segmentation
#### mask R-CNN
![mask R-CNN_1.png](https://i.imgur.com/sqbPeWy.png)

将图像输入卷积网络和训练好的边框标注网络，得到候选边框后，把候选框投射到卷积特征图上，接下来想做的是对每一个候选框预测出一个分割区域。
将候选框对齐到特征后，对应的候选框就有了正确的尺寸，这样得到了两个分支，一个是预测类别分数，告诉我们候选框对应哪个分类，也预测边界框坐标；另一个分支对输入候选框的像素进行分类，确定是不是属于某个目标物体。
Mask R-CNN是在原有的Fast R-CNN的添加。Mask R-CNN结合了所讲的所有内容，还可以做人体姿态检测。