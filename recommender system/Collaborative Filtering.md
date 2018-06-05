### 1.0 基于用户的协同过滤算法
1.找到和目标用户兴趣相似的用户集合。

![user_based_cf_2.png](https://i.imgur.com/REOZtSV.png)

给定用户u和用户v，令N(u)表示用户u曾经有过正反馈的物品集合，令N(v)为用户v曾经有过正反馈的物品集合。

2.找到这个集合中的用户喜欢的，且目标用户没有听说过的物品推荐给目标用户。

![user_based_cf_1.png](https://i.imgur.com/gN81NJ9.png)

其中，S(u,K)包含和用户u兴趣最接近的K个用户，N(i)是对物品i有过行为的用户集合，w<sub>uv</sub>是用户u和用户v的兴趣相似度， r<sub>vi</sub>代表用户v对物品i的兴趣

缺陷：
1. 首先，随着网站的用户数目越来越大，计算用户兴趣相似度矩阵将越来越困难，其运算时间复杂度和空间复杂度的增长和用户数的增长近似于平方关系。
2. 其次，基于用户的协同过滤很难对推荐结果作出解释。

### 2.0 基于物品的协同过滤算法
#### 2.1 基础算法
步骤：
1.计算物品之间的相似度。

![item_based_cf_1.png](https://i.imgur.com/MxeZRUT.png)

分母|N(i)|是喜欢物品i的用户数</br>
分子|N(i)intersectionN(j)|是同时喜欢物品i和物品j的用户数。

![item_based_cf_2.png](https://i.imgur.com/UvPeWQn.png)

相似度计算：

```python
def itemSimilarity(user_items):
    C = dict()
    N = dict()
    for user,items in user_items.items():
        for i in items.keys():
            N.setdefault(i, 0)
            N[i] += 1
            C.setdefault(i, {})
            for j in  items.keys():
                if i == j:
                    continue
                C[i].setdefault(j, 0)
                C[i][j] += 1
    W = dict()
    for i,related_items in C.items():
        W.setdefault(i, {})
        for j,cij in related_items.items():
            W[i][j] = cij/(sqrt(N[i]*N[j]))
    return W,C,N
```

2.根据物品的相似度和用户的历史行为给用户生成推荐列表。

![item_based_cf_3.png](https://i.imgur.com/mUHcCOG.png)

N(u)是用户喜欢的物品的集合，S(j,K)是和物品j最相似的K个物品的集合，w<sub>ji</sub>是物品j和i的相似度，r<sub>ui</sub>是用户u对物品i的兴趣。

代码实现：

```python
def recommend(user_items, iteSimilarity, user, k=3, N=10):
    rand = dict()
    print("user_items=%s",user_items)
    # 用户收藏的喜欢商品：action_items={'a': 1, 'b': 1, 'd': 1}
    action_items = user_items[user]
    for item,score in action_items.items():
        # j,wj={'b': 0.4082482904638631, 'd': 0.7071067811865475}
        for j,wj in sorted(itemSimilarity[item].items(), key=lambda x:x[1], reverse=True)[0:k]:
        #与item最相似的k个商品
            if j in action_items.keys(): # 不在用户收藏的喜欢商品
                continue
            rand.setdefault(j, 0)
            rand[j] += score*wj
            print(j,wj,rand[j])
    return dict(sorted(rand.items(), key=lambda x:x[1], reverse=True)[0:N])
```

基于物品的协同过滤算法的例子

![item_based_cf_4.png](https://i.imgur.com/f0fs5ir.png)

#### 2.2 物品相似度的归一化
![user_based_cf_3.png](https://i.imgur.com/2M8srxk.png)

其实，归一化的好处不仅仅在于增加推荐的准确度，它还可以提高推荐的覆盖率和多样性。

#### 2.3 UserCP和ItemCF的综合比较
原始ItemCF算法的覆盖率和新颖度都不高。

物品相似度公式改进：

![user_based_cf_4.png](https://i.imgur.com/PqJljXx.png)

其中α属于[0.5,1]。通过提高α，可以惩罚热门的j。</br>
通过这种方法可以在适当牺牲准确率和召回率的情况下显著提升结果的覆盖率和新颖性。

### 2.5 隐语义模型
隐含语义分析技术从诞生到今天产生了很多著名的模型和方法，其中和该技术相关且耳熟能详的名词有pLSA、 LDA、隐含类别模型（latent class model）、隐含主题模型（latent topic model）、矩阵分解（matrix factorization）。

LFM通过如下公式计算用户u对物品i的兴趣：

![lfm_1.png](https://i.imgur.com/I9kpvef.png)

p<sub>u,k</sub>度量了用户u的兴趣和第k个隐类的关系，q<sub>i,k</sub>度量了第k个隐类和物品i之间的关系。

采样时应该遵循以下原则：</br>
1. 对每个用户，要保证正负样本的平衡（数目相似）。
2. 对每个用户采样负样本时，要选取那些很热门，而用户却没有行为的物品。


1. [CollaborativeFilter](https://github.com/ScofieldShen/MLRep/tree/ce0ccdfb9939e70c183504ee4be59ccba235cb47/ML/CollaborativeFilter)
2. [推荐引擎初探](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy1/index.html)
3. [深入推荐引擎相关算法 - 协同过滤](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/index.html)