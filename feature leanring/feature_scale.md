### 1.0 特征缩放
#### 1.1.1 调节比例
![调节比例](https://i.imgur.com/GfRSDjV.png)

- 优点：预估输出相对稳定
- 缺点：如果输出特征中有异常值，那么特征缩放就会比较棘手（最大值最小值可能是极端值）

###### 应用
```
from sklearn.preprocessing import MinMaxScaler
import numpy
#这里numpy数组中的是特征，因为此处特征只有一个，所以看起来是这样的
#因为这里应该作为一个浮点数进行运算，所以数字后面要加.
weights = numpy.array([[115.],[140.],[175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
print rescaled_weight
```

#### 1.1.2 标准化
![标准化](https://i.imgur.com/9IDUEHR.png)

特征标准化使每个特征的值有平均值(zero-mean)和单位方差(unit-variance)。

#### 1.1.3 对数log
```
# 使用自然对数缩放数据
log_data = np.log(data)
```
如果数据**不是正态分布**的，尤其是数据的**平均数和中位数相差很大**的时候（表示数据非常歪斜）。这时候通常用一个**非线性**的缩放是很合适的。
尤其是对于金融数据。一种实现这个缩放的方法是使用**Box-Cox** 变换，这个方法能够计算出能够最佳减小数据倾斜的指数变换方法。一个比较简单的并且在大多数情况下都适用的方法是使用自然对数。

##### 其他的转换
![经济学转换方式_1.jpg](https://i.imgur.com/PevGIqH.jpg)

#### 1.2 Box-Cox
![Box-Cox.jpg](https://i.imgur.com/RYBEGyi.jpg)

###### 没有Box-Cox变换的回归
![before_Box-Cox.jpg](https://i.imgur.com/QiRZGTK.jpg)

###### Box-Cox变换之后的回归
![after_Box-Cox.jpg](https://i.imgur.com/cDEP7y3.jpg)


#### 1.3.1 哪些机器学习算法会受到特征缩放的影响？
- SVM(rbf)计算最大距离时就是这种情况。如果我们把某一点增大至其他点的两倍，那么它的数值也会扩大一倍
- K-均值聚类也是。计算各数据点到集群中心的距离