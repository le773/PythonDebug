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

#### 1.2.1 哪些机器学习算法会受到特征缩放的影响？
- SVM(rbf)计算最大距离时就是这种情况。如果我们把某一点增大至其他点的两倍，那么它的数值也会扩大一倍
- K-均值聚类也是。计算各数据点到集群中心的距离