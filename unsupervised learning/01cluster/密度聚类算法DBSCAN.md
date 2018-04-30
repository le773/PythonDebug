### 1.0 算法流程简介

![dbscan_2.jpg](https://i.imgur.com/VByQOmv.jpg)

算法的主要目标是相比基于划分的聚类方法和层次聚类方法:
1. 需要更少的领域知识来确定输入参数；
1. 发现任意形状的聚簇；
1. 在大规模数据库上更好的效率。

算法流程能够将足够高密度的区域划分成簇，并能在具有噪声的空间数据库中发现任意形状的簇。

### 2.0 算法流程
算法流程（伪代码）如下：
```
DBSCAN(D, eps, MinPts)
   C = 0                                          
   for each unvisited point P in dataset D        
      mark P as visited                           
      NeighborPts = regionQuery(P, eps)      //计算这个点的邻域    
      if sizeof(NeighborPts) < MinPts             
         mark P as NOISE                          
      else                                        
         C = next cluster                   //作为核心点，根据该点创建一个新类簇
         expandCluster(P, NeighborPts, C, eps, MinPts)   //根据该核心点扩展类别
          
expandCluster(P, NeighborPts, C, eps, MinPts)
   add P to cluster C                            //扩展类别，核心点先加入
   for each point P' in NeighborPts                    
      if P' is not visited
         mark P' as visited                              
         NeighborPts' = regionQuery(P', eps)    //如果该点为核心点，则扩充该类别
         if sizeof(NeighborPts') >= MinPts
            NeighborPts = NeighborPts joined with NeighborPts'
      if P' is not yet member of any cluster   //如果邻域内点不是核心点，并且无类别，比如噪音数据，则加入此类别
         add P' to cluster C
          
regionQuery(P, eps)                                       //计算点P的邻域
   return all points within P's eps-neighborhood
```

这里的`regionQuery`函数的作用计算点的邻域，是比较耗费时间的操作，不进行任何优化时，算法的时间复杂度是`O(N**2)`，通常可利用`R-tree`，`k-d tree`, `ball tree`索引来加速计算，将算法的时间复杂度降为`O(Nlog(N))`。

### 3.0 算法优缺点

缺点：`DBSCAN`使用了统一的`eps`邻域值和`Minpts`值，在类中的数据分布密度不均匀时，`eps`较小时，密度小的`cluster`会被划分成多个性质相似的`cluster`；`eps`较大时，会使得距离较近且密度较大的`cluster`被合并成一个`cluster`。在高维数据时，因为维数灾难问题，`eps`的选取比较困难。

优点：能发现任意形状的聚簇，聚类结果几乎不依赖于结点遍历顺序，能够有效的发现噪声点。

![dbscan_1.jpg](https://i.imgur.com/V8HDZzX.jpg)

详细介绍见：[聚类算法第三篇-密度聚类算法DBSCAN](https://zhuanlan.zhihu.com/p/23504573)