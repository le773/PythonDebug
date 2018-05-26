### 四分位距 IQR(interquartile range)

##### 数据的75th分位点 - 数据的25th分位点间距
![IQR_1](http://datapigtechnologies.com/blog/wp-content/uploads/2014/01/011614_0525_Highlightin4.png)

异常值：
values below (Quartile 1) – IQR)
values above (Quartile 3) + IQR)

##### 1.5倍(数据的75th分位点 - 数据的25th分位点间距)
![IQR_2](http://datapigtechnologies.com/blog/wp-content/uploads/2014/01/011614_0525_Highlightin6.png)

异常值：
values below (Quartile 1) – (1.5 × IQR)
values above (Quartile 3) + (1.5 × IQR)

### 实例
```
import numpy as np
# 可选：选择你希望移除的数据点的索引
outliers  = []

# 对于每一个特征，找到值异常高或者是异常低的数据点
for feature in log_data.keys():
    
    # TODO：计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data[feature], 25, axis=0)
    
    # TODO：计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature], 75, axis=0)
    
    # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
    step = 1.5 * (Q3 - Q1)
    
    # 显示异常点
    tmp_beyond = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    outliers += tmp_beyond.index.tolist() # 获取索引

# 如果选择了的话，移除异常点 并重新索引
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
good_data.head()
```
[Highlighting Outliers in your Data with the Tukey Method](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/ "Highlighting Outliers in your Data with the Tukey Method")