### 1.0 GaussianMixture
#### 1.0.1 GaussianMixture 概念
是用高斯概率密度函数（正态分布曲线）精确地量化事物，将一个事物分解为若干的基于高斯概率密度函数（正态分布曲线）形成的模型。通俗点讲，无论观测数据集如何分布以及呈现何种规律，都可以**通过多个单一高斯模型的混合进行拟合**。

![GaussianMixture](https://i.imgur.com/cPzypaa.png)

如图，数据集明显分为两个聚集核心，我们通过两个单一的高斯模型混合成一个复杂模型来拟合数据。这就是一个混合高斯模型。

#### 1.0.2 GMM、EM算法
EM（Expectation Maximization Algorithm：期望最大化）

![gmm_em.png](https://i.imgur.com/YrrWTpv.jpg)

#### 1.0.3 代码用例
```
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

n_range = range(2,11)
result3 = []
for i in n_range:
    # TODO：在降维后的数据上使用你选择的聚类算法
    clusterer = GaussianMixture(n_components=i, random_state=12)
    clusterer.fit(reduced_data)
    # TODO：预测每一个点的簇
    preds = clusterer.predict(reduced_data)

    # TODO：找到聚类中心
    centers = clusterer.means_

    # TODO：预测在每一个转换后的样本点的类
    sample_preds = clusterer.predict(pca_samples)

    # TODO：计算选择的类别的平均轮廓系数（mean silhouette coefficient）
    score = silhouette_score(reduced_data, preds)
    result3.append(score)
print result3
```

相同的数据，K-means,GMM,GaussianMixture的对比

![cluster_silhouette_score](https://i.imgur.com/35Jptkd.png)