### 01 hist可视化
```
l_subscribers = []
l_customers = []
l_subscribers, l_customers = filter_bike_time(city)
# x轴间隔
plt.hist(l_customers, bins=[x for x in range(0, 80, 5)], range=(min(l_customers), 75))
plt.title('Distribution of Trip Durations')
plt.xlabel('{} {} Duration (m)'.format(city, 'customers'))
plt.show()
# x轴间隔
plt.hist(l_subscribers, bins=[x for x in range(0, 80, 5)], color='green',range=(min(l_subscribers), 75))
plt.title('Distribution of Trip Durations')
plt.xlabel('{} {} Duration (m)'.format(city, 'subscribers'))
plt.show()
```
![distribution_of_trip_durations_1](https://i.imgur.com/VewWe2W.png)

![pandas bar](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/67065/1514970688/TIM%E6%88%AA%E5%9B%BE20180103162043.png)

### 02 2行1列折线图
```
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=5)
l2 = cross_val_score(regressor, boston.data, boston.target, cv=10)

max_depth = regressor.get_params()['max_depth']
# print(max_depth)


print(len(boston.data), len(boston.data[0]))

print(l2)

print(boston.data[0])
l1 = [ x + 1 for x in l2]

fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
# plt.plot(np.arange(len(l2)), boston.data[0], 'go-', label='true value')
t1, = ax1.plot(np.arange(len(l2)), l2, 'ro-', label = 'line')

ax2 = fig1.add_subplot(212)
t2, = ax2.plot(np.arange(len(l2)), l1, 'go-', label = 'parabola' ,color = 'gray', linewidth = 1.0, linestyle = '--')
# plt.title('score: %f' % score)
plt.legend(handles = [t1, t2,], labels=['t1', 't2'])
plt.show()
```

![折线图](https://i.imgur.com/dFtkb5H.png)

### 03 hist 柱形图 设置
```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams

# 添加label
fig1 = plt.figure(3)

bar1 =plt.bar(left = 0.2, height = 1, color='r', width = 0.2, align="center",yerr=0.000001)
bar2 =plt.bar(left = 0.6, height = 1.5, color='g', width = 0.2, align="center",yerr=0.000001)
bar3 =plt.bar(left = 1, height = 0.2, color='b', width = 0.2, align="center",yerr=0.000001)

plt.xticks((0.2, 0.6, 1),('first','second', 'three'))
plt.yticks((0.2, 0.6, 1),('low','meida', 'high'))
plt.title('f hist')

# 为每个条形设置数字及其位置
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.1 * height, 'v:%s' % float(height), color='black', rotation=90)

autolabel(bar1)
autolabel(bar2)
autolabel(bar3)

plt.legend(labels=['bar1', 'bar2', 'bar3'])

plt.show()
```
![直方图_3.png](https://i.imgur.com/u3um0rQ.png)