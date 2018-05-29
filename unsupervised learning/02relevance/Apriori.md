## Apriori
### 0.0 Apriori
频繁项集：满足最小支持度的项集；</br>
关联规则（association rules）：暗示两种物品之间可能存在很强的关系；</br>
支持度（support）：被定义为数据集中包含该项集的记录所占的比例；</br>
可信度或置信度（confidence）：是针对关联规则来定义的；

### 1.0 支持度的计算
#### 1.1.1 原理
设数据集(D)：[set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]；</br>
候选项集列表(C1)：[frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]；</br>
frozenset([1])在D中出现的次数为2，</br>
同理其他：frozenset([4]): 1, frozenset([5]): 3, frozenset([2]): 3, frozenset([3]): 3</br>
支持度=frozenset([1])出现的次数/数据集(D)的长度

#### 1.1.2 代码实现:生成频繁项集
```python
# 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
def scanD(D, Ck, minSupport):
    """scanD（计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度 minSupport 的数据）

    Args:
        D 数据集
        Ck 候选项集列表
        minSupport 最小支持度
    Returns:
        retList 支持度大于 minSupport 的集合
        supportData 候选项集支持度数据
    """
    # D = [set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]
    # C1:  [frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]
    # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # s.issubset(t)  测试是否 s 中的每一个元素都在 t 中
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    print 'ssCnt:',ssCnt
    numItems = float(len(D)) # 数据集 D 的数量
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    print 'scanD result >> '
    print 'retList:', retList
    print 'supportData:', supportData
    return retList, supportData
```
### 2.0 生成关联规则
Apriori原理是说如果某个项集是频繁的，那么它的所有子集也是频繁的。更常用的是它的逆否命题，即如果一个项集是非频繁的，那么它的所有超集也是非频繁的。

![genrule_1.png](https://i.imgur.com/AHPr1HU.png)

#### 2.1.1 可信度的计算
```python
# 计算可信度（confidence）
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """calcConf（对两个元素的频繁项，计算可信度，例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件）

    Args:
        freqSet 频繁项集中的元素，例如: frozenset([1, 3])
        H 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的空数组
        minConf 最小可信度
    Returns:
        prunedH 记录 可信度大于阈值的集合
    """
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
	# 假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]，
	# 那么现在需要求出 frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
    for conseq in H: 
        print 'confData=', freqSet, H, conseq, freqSet-conseq
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 先推导关联规则：123->4
		# 支持度定义: a -> b = support(a | b) / support(a). 
		# 假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，
		# 那么 frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b) / support(a) 
		# = supportData[freqSet]/supportData[freqSet-conseq]
        # = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print freqSet-conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    print "prunedH:>",prunedH 
	# prunedH中的item元素通过aprioriGen生成len(item)+1的子集D2，再次从freqSet发掘D2存在的关联
    return prunedH
```

#### 2.1.2 生成关联规则
```python
# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    """generateRules

    Args:
        L 频繁项集列表
        supportData 频繁项集支持度的字典
        minConf 最小置信度
    Returns:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """
    print 'L 频繁项集列表:',L
    bigRuleList = []
    # 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], 
	# [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], 
    # [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        print "i=>",i, L[i]
        for freqSet in L[i]:
            # 假设：freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])]
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if (i > 1):
            # freqSet = L[>=2][0] = [frozenset([2, 3, 5])],
            # H1是freqSet的元素集合 [frozenset([2]), frozenset([3]), frozenset([5])]
            # 此时需要生成规则：freqSet 和 H1.length+1元素的关系
                calcConf(freqSet, H1, supportData, bigRuleList, minConf) # 应该先考虑123->1234的情形
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            # 当第一次生成关联规则时：freqSet=L[1][0]=frozenset([1, 3]) 
            # H1是freqSet的元素集合[frozenset([1]), frozenset([3])]
            else:
                print 'calcConf: freqSet>', freqSet, 'H1>', H1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
```

#### 2.1.3 递归计算频繁项集的规则
```python
# 递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """rulesFromConseq

    Args:
        freqSet 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的数组
        minConf 最小可信度
    """
    # freqSet= frozenset([1, 3]), H=frozenset([1])
    # H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
    # 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
    # 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
    # 那么 m = len(H[0]) 的递归的值依次为 1 2
    # 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，没必要再计算 freqSet 与 H[0] 的关联规则了。
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        print 'freqSet******************', len(freqSet), m + 1, freqSet, H, H[0]
        # 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = aprioriGen(H, m+1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print 'Hmp1=', Hmp1
        print 'freqSet=',freqSet
        print 'len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet)
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            print '----------------------', Hmp1
            print len(freqSet),  len(Hmp1[0]) + 1
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
```

#### 2.2 关联规则的推导过程总结
情况一，当frozenset的长度为2：
1. 假设freqSet={1,2}，则H1={1},{2}；
2. 计算置信度：
freqSet-H1.element -> freqSet，即{2}->{1,2}、{1}->{1,2}

情况二，当frozenset的长度大于2：
1. 假设freqSet={1,2,3,4}时，则H1={1},{2},{3},{4}；
2. 首先计算置信度：
freqSet-H1.element -> freqSet，即{1,2,3}->{1,2,3,4}、{2,3,4}->{1,2,3,4}、..；
3. 递归计算频繁项集</br>
3.a 首先通过aprioriGen计算H<sub>m+1</sub>，H2={1,2}、{3,4}、..</br>
3.b 计算置信度：</br>
freqSet-H1.element -> freqSet，即{1,2}->{1,2,3,4}、{1,3}->{1,2,3,4}、..；</br>
3.c当Len(H<sub>m+1</sub>[0])+1<Len(freqSet)，则(3.a)；否则结束

### 3.0 优劣
优点：
1. 简单，易理解
2. 数据要求低

缺点：
1. 在每一步产生候选集时循环产生的组合过多，没有排除不应该参与组合的元素。
2. 每次计算项集的支持度时，都对数据库中的全部记录进行了一遍扫描比较，如果是一个大型的数据库时，这种扫描会大大增加计算机的I/O开销。


参考：

1. [使用Apriori算法和FP-growth算法进行关联分析](http://www.cnblogs.com/qwertWZ/p/4510857.html)