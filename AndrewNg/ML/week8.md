## 02 motivation
### 02.01 motivation I:Data Compression

###### 2D->1D
![motivation_data_compression_1](https://i.imgur.com/eY72lvl.png)

###### 3D->2D
![motivation_data_compression_2](https://i.imgur.com/LgBaNCR.png)

1. 占用更少的计算机内存和硬盘空间
2. 给算法提速度

### 02.01 motivation II:Visualization
降维以可视化。
![motivation_visualization_1](https://i.imgur.com/DjsG4qX.png)


## 03 principal component analysis
### 03.01 principal component analysis problem formulation
![pca_problem_formulation_1](https://i.imgur.com/AGA6WSc.png)
PCA 所做的就是:寻找一个低维的面,数据投射在上面,使得这些蓝色小线段的平方和达到最小值;这些蓝色线段的长度时常被叫做投影误差，所以PCA所做的就是**寻找一个投影平面，对数据进行投影，使得这个能够最小化**;
另外,在应用PCA之前,通常的做法是:先进行**均值归一化**和**特征规范化**，使得特征`x1`和`x2`均值为0，数值在可比较的范围之内。

#### 03.01.01 pca is not linear regression
![pca_is_not_linear_regression_1](https://i.imgur.com/5TXTBHt.png)
线性回归：在给定某个特征输入x的情况下，预测某个变量y的数值，因此，对于线性回归，是拟合一条直线来最小化点和直线之间的平方误差。
1. pca：1. 是点到直线的距离最小；
pca的目的是：寻找一个低维的平面对数据进行投影，以便最小化投影误差的平方，最小化每个点与投影后的对应点的距离的平方值。
2. 所有的特征被同等的对待；
PCA中只有特征没有标签数据y，LR中既有特征样本也有标签数据。

### 03.02 pca algorithrm
###### 03.02.01 数据预处理
![data_preprocessing_1](https://i.imgur.com/0rx42q0.png)

###### 03.02.02 pca算法
![pca_algorithm_1](https://i.imgur.com/1BLVZcj.png)

总结：1. 数据均值预处理; 2. 计算协方差矩阵，选取topK 3.降维 4. 构建新的数据集

## 04 Applying PCA
### 04.01 Reconstruction from Compressed Representation
从压缩的数据还原数据
![reconstruction_from_compressed_representation_1](https://i.imgur.com/oqCsdV8.png)
```
reconMat = (lowDDataMat * redEigVects.T) + meanVals
# redEigVects topK特征
# lowDDataMat 降维后数据集
```

### 04.02 Choosing the Number of Principal Compoents
- Average squared projection error
![average_squared_projection_error_1](https://i.imgur.com/bmgsms2.png)

- total variation in the data
![total_variation_1](https://i.imgur.com/CpT5L87.png)
含义：平均来看训练样本距离零向量多远。

![chooseK](http://img.blog.csdn.net/20160611214044382)

当表达式小于0.01时，意味着0.99的方差(差异性)被保留

##### 04.02.02 Choosing k
![pca_algorithm_choosek_1](https://i.imgur.com/cuDFlaI.png)

![pca_algorithm_choosek_2](https://i.imgur.com/5hain1Q.png)

### 04.03 Advice for Applying PCA
![supervised_learning_speedup_1](https://i.imgur.com/gFPeab0.png)
交叉验证集、测试集均可使用pca降维的数据。

#### 04.03.02 PCA的应用 
- 数据压缩
压缩数据，减小存储空间和内存空间
加速学习算法
- 可视化

#### 04.03.03 pca不适合用作防止过拟合
![pac_not_use_to_prevent_overfitting_1](https://i.imgur.com/AxlMeJT.png)
PCA可以减少特征的数量（维数），所以在理论上讲可以防止过拟合问题，但是并不是最好的方法。最好的方法还是利用规则化参数对算法防止过拟合。

#### 04.03.04 pca is sometimes used where it shouldn't be
![pca is sometimes used where it shouldn't be](http://img.blog.csdn.net/20160611215319964)

并不是所有的问题都是要对原始数据进行PCA降维，
- 首先应看在不使用PCA的情况下算法的运行情况，
- 如果未达到所期望的结果，再考虑PCA对数据进行降维。