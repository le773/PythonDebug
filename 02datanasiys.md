### 0.0	数据分析流程
提出数据
整理数据
探索数据
得出结论
进行交流
### 1.0 数据分析过程
#### 1.1.1	在 Pandas 中选择多个范围 
选择均值数据框的列非常简单，因为需要选择的列都在一起（`id`、`diagnosis` 以及均值列）。现在选择标准误差或最大值的列时，出现了一点问题。`id` 和  `diagnosis` 和所需的其它列是分开的，无法在一个范围内指定全部。

首先，尝试自己创建标准误差数据框，了解为什么只有 `loc` 和 `iloc` 是分开的。然后，打开 [stackoverflow 链接](https://stackoverflow.com/questions/41256648/select-multiple-ranges-of-columns-in-pandas-dataframe) 学习如何在 Pandas 中选择多个范围，然后再次尝试。顺便说一下，我自己是在 google 中搜索 “如何选择多个范围 df.iloc” 时发现这个链接的。
```
# 选择581到583行
df.iloc[581:583]
```
###### csv简介
- csv仅存储原始数据
- csv是纯文本类型

```python
import pandas as pd
df = pd.read_csv('cancer_data.csv')
# 指定分隔符
df = pd.read_csv('cancer_data.csv', sep=';')
# 读取文本
df = pd.read_table('smsspamcollection/SMSSpamCollection',header=None,names=columns,sep='\t')

# 创建最大值数据框
max_df_cancer_data_edited=df.loc[:,'radius_max':'fractal_dimension_max']
# 查看前几行，确认是否成功
max_df_cancer_data_edited.head()

# 返回数据框维度的元组(行列数量)
df.shape

# 返回列的数据类型
df.dtypes

# 虽然供诊断的数据类型是对象，但进一步的
# 调查显示，它是字符串
type(df['diagnosis'][0])

# 显示数据框的简明摘要，
# 包括每列非空值的数量
# 统计数据缺失
df.info()

# 返回每列数据的有效描述性统计
df.describe()
count    6497.000000	计数
mean       10.491801	平均值
std         1.192712	标准差
min         8.000000	最小值
25%         9.500000	第一个四分位
50%        10.300000	中位数
75%        11.300000	第三个四分位
max        14.900000	最大值
Name: alcohol, dtype: float64

# 但是也可以指定你希望返回的行数
df.head(20)

# `.tail()` 返回最后几行，但是也可以指定你希望返回的行数
df.tail(2)

# 查看每列的索引号和标签
for i, v in enumerate(df.columns):
    print(i, v)
```

panda存储字符串的指针，而非字符串本身，所以列类型为object

**数据冗余、数据类型不正确、数据丢失**和数据不准确都是我们在分析数据之前需要解决的问题。

#### 1.2.1数据缺失
使用平均值填充
```
mean = df['view_duration'].mean()
df['view_duration'].fillna(mean,inplace=True) #inplace原地修改
```
#### 1.2.2数据冗余

```
# 删除不需要的列之后，在统计冗余的行
df.duplicated()
sum(df.duplicated())#统计冗余的行数
#df.duplicated().sum()
df.drop_duplicates(inplace=True)#删除冗余的行并填充到原始数据集
# drop_duplicates subset
```
#### 1.2.3数据类型错误
string转为datetime
```
df['timestamp']=pd.to_datetime(df['timestamp'])
```
即使保存到原始数据，下次打开时候，文件仍然将数据读为为string类型

#### 1.3.1 重命名列
由于之前修改了数据集，使其仅包括肿瘤特征的均值，因此每个特征末尾好像不需要 "_mean" 。而且，稍后输入分析还要多耗费时间。我们现在想一些要分配给列的新标签。
方法1
```
# 从列名称中移除 "_mean"
new_labels = []
for col in df.columns:
    if '_mean' in col:
        new_labels.append(col[:-5])  # 不包括最后 6 个字符
    else:
        new_labels.append(col)

# 列的新标签
new_labels

# 为数据框中的列分配新标签
df.columns = new_labels

# 显示数据框的前几行，确认更改
df.head()

# 将其保存，供稍后使用
df.to_csv('cancer_data_edited.csv', index=False)
```
方法2
```
new_labels = list(df.columns)
new_lables[4] = 'new name'
df.columns = new_lables

df.head()
```

#### 1.4.1 使用 Pandas 绘图

`%matplotlib inline`
使用`%matplotlib`命令可以将`matplotlib`的图表直接嵌入到`Notebook`之中，或者使用指定的界面库显示图表，它有一个参数指定`matplotlib`图表的显示方式。**inline表示将图表嵌入到Notebook中**
```
df.hist(figsize=(8,8)); #查看该csv的直方图
#;隐藏不必要的输出
df['column1'].hist(figsize=(8,8));	#查看该列column1 csv的直方图
df['column1'].plot(kind='hist');	#查看该列column1 csv的直方图
df['column1'].value_counts()		#该列column1每个唯一值的数量
df['column1'].value_counts().plot(kind='bar');
#该列column1每个唯一值的数量绘制带间隔的直方图
df['column1'].value_counts().plot(kind='pie',figsize=(8,8));
#该列column1每个唯一值的数量绘制饼图
pd.plotting.scatter_matrix(df,figsize=(8,8))#该df所有变量绘制散列图
df.plot(x='column1',y='column2',kind='scatter');#该df指定x,y变量绘制散列图
df['column1'].plot(kind='box')#查看该df列column1 csv的箱线图
```
#### 1.4.2 箱形图
箱形图（Box-plot）又称为盒须图、盒式图或箱线图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域也经常被使用，常见于品质管理。
![箱线图](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=28928945f1deb48fef64a98c9176514c/0b55b319ebc4b74596c1a432cdfc1e178a8215b8.jpg)

选取diagnosis为M的数据组成新的表
```
df_m = df[df['diagnosis'] == 'M']
df_m.head()
```
#### 1.4.3 拆分数据，并按照指定的行显示
```
# 按照收入分割数组
#df_a = df[df['income'] == ' >50K']
# 获取索引
#index = df_a['education'].value_counts().index
# 按照指定索引绘图
#df_a['education'].value_counts()[index].plot(kind='bar');
# 饼图
#df_a['education'].value_counts()[index].plot(kind='bar', figsize=(8,8));
```

#### 1.4.4 添加列
```
# 为红葡萄酒数据框创建颜色数组
red_row_len = red_df.shape[0]
# 每一行添加颜色字段
color_red = np.repeat('red', red_row_len)
# 添加到数组
red_df['color'] = color_red
red_df.head()
```

#### 1.4.5 附加数据集合
```
# 附加数据框
wine_df = red_df.append(white_df, ignore_index=True)
wine_df.to_csv('winequality_edited.csv', index=False)
# 查看数据框，检查是否成功
wine_df.head()
```
###### refs: Merge join and concatenate
```
http://pandas.pydata.org/pandas-docs/stable/merging.html#merging-concatenation
```

重命名列

索引不支持可变操作
实例参考：方法2

```
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html
```
#### 1.5.1 Pandas GroupBy
```
# 单个值
df.groupby(['quality']).mean()
# 多个值
df.groupby(['quality','color']).mean()

# groupby分组列不作为索引
df.groupby(['quality','color'], as_index=False).mean()

# groupby分组列不作为索引,查看单个列的值
df.groupby(['quality','color'], as_index=False)['ph'].mean()
```

#### 1.5.2 根据酸度水平的评定酸度级别
```
# Load `winequality_edited.csv`
import pandas as pd
df = pd.read_csv('winequality_edited.csv')
df.head()

# 用 Pandas 描述功能查看最小、25%、50%、75% 和 最大 pH 值
df['pH'].describe()

# 对用于把数据“分割”成组的边缘进行分组
# 用刚才计算的五个值填充这个列表
bin_edges = [2.720000, 3.110000, 3.210000, 3.320000, 4.010000]

# 四个酸度水平组的标签
# 对每个酸度水平类别进行命名
bin_names = ['high', 'medium on the high side', 'medium', 'low']

# 创建 acidity_levels 列
df['acidity_levels'] = pd.cut(df['pH'], bin_edges, labels=bin_names)

# 检查该列是否成功创建
df.head()

# 哪个水平的酸度获得最高的平均评级？
df.groupby(['acidity_levels'], as_index=False)['quality'].mean()

# 各个年龄的爽约率 (自动排序)
df[df['xxx'] == 1]['acidity_levels'].value_counts().plot(kind='bar', rot=0, figsize=(8, 3));

# acidity_levels 分布
df['acidity_levels'].hist(figsize=(8,8),bins=16);
```

#### 1.6.1 query
```
# selecting malignant records in cancer data
df_m = df[df['diagnosis'] == 'M']
df_m = df.query('diagnosis == "M"')

# selecting records of people making over $50K
df_a = df[df['income'] == ' >50K']
df_a = df.query('income == " >50K"')
```
#### 1.6.2 实例
```
# 加载 `winequality_edited.csv`
import pandas as pd
df = pd.read_csv('winequality_edited.csv')
df.head()

# 获取酒精含量的中位数
df['alcohol'].describe()

# 选择酒精含量小于平均值的样本
low_alcohol = df.query('alcohol < 10.300000')
print(low_alcohol['quality'].count())

# 选择酒精含量大于等于平均值的样本
high_alcohol = df.query('alcohol >= 10.300000')
print(high_alcohol['quality'].count())

# 确保这些查询中的每个样本只出现一次
num_samples = df.shape[0]
print(num_samples)
num_samples == low_alcohol['quality'].count() + high_alcohol['quality'].count()
# 应为真

# 获取低酒精含量组的平均质量评分
low_alcohol['quality'].mean()

# 高酒精含量组的平均质量评分
high_alcohol['quality'].mean()
```

#### 1.6.3 绘制直方图
```
# 加载 `winequality_edited.csv`
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

colorss=['green','red']
color_means=df.groupby(['color'])['quality'].mean()
color_means.plot(kind='bar', title='avg', color=colorss, alpha=.7)
plt.xlabel('color', fontsize=18)
plt.ylabel('quality', fontsize=18)
```
#### 1.6.4 指定x轴标签
为：'a', 'b', 'c'
```
plt.bar([1, 2, 3], [224, 620, 425], tick_label=['a', 'b', 'c'])
plt.title('Some Title')
plt.xlabel('Some X Label')
plt.ylabel('Some Y Label');
```

#### 1.6.5 实例
```
# 用查询功能选择每个组，并获取其平均质量
median = df['alcohol'].median()
low = df.query('alcohol < {}'.format(median))
high = df.query('alcohol >= {}'.format(median))

mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()


# 用合适的标签创建柱状图
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Average Quality Rating');
```

#### 1.6.6 实例:为红葡萄酒条柱高度和白葡萄酒条柱高度创建数组
```
# 01
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')

wine_df = pd.read_csv('winequality_edited.csv')

# 02
# 获取每个等级和颜色的数量
color_counts = wine_df.groupby(['color', 'quality']).count()['pH']
color_counts

# 获取每个颜色的总数
color_totals = wine_df.groupby('color').count()['pH']
color_totals

# 03
# 将红葡萄酒等级数量除以红葡萄酒样本总数，获取比例
red_proportions = color_counts['red'] / color_totals['red']
red_proportions

# 将白葡萄酒等级数量除以白葡萄酒样本总数，获取比例
white_proportions = color_counts['white'] / color_totals['white']
white_proportions
# 04
ind = np.arange(len(red_proportions))  # 组的 x 坐标位置
width = 0.35  

# 04绘制条柱
red_bars = plt.bar(ind, red_proportions, width, color='r', alpha=.7, label='Red Wine')
white_bars = plt.bar(ind + width, white_proportions, width, color='w', alpha=.7, label='White Wine')

# 05
# 标题和标签
plt.ylabel('Proportion')
plt.xlabel('Quality')
plt.title('Proportion by Wine Color and Quality')
locations = ind + width / 2  # x 坐标刻度位置
labels = ['3', '4', '5', '6', '7', '8', '9']  # x 坐标刻度标签
plt.xticks(locations, labels)

# 图例
plt.legend()
```

#### 1.7.1 zip
```
m = [[1,2,3], [4,5,6], [7,8,9]]
n = [[2,2,2], [3,3,3], [4,4,4]]
p = [[2,2,2], [3,3,3,]
zip(m, n)将返回
([1, 2, 3], [2, 2, 2]), 
([4, 5, 6], [3, 3, 3]), 
([7, 8, 9], [4, 4, 4])

zip(m, p)将返回
([1, 2, 3], [2, 2, 2]), 
([4, 5, 6], [3, 3, 3])
([7,8,9])
```
参考
[Python中的zip()与*zip()函数详解](https://www.cnblogs.com/waltsmith/p/8029539.html)

#### 1.7.2 保留小数
```
保留浮点数的小数点。

     如保留小数点后两位。

     num = 9.2174

     new_num = round( num , 2 )

     则new_num = 9.22    (四舍五入）
```

#### 1.7.3 去除语句中的标点符号
```
line.translate(str.maketrans('', '', string.punctuation))
```

----------
### 03 数据分析 案例研究2
#### 03.01 丢弃多余的列
```
# 从 2008 数据集中丢弃列
df_08.drop(['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'], axis=1, inplace=True)

# 确认更改
df_08.head(1)
```
#### 03.02 规范列名：大写改小写 空格改下划线
```
# 在 2018 数据集中用下划线和小写标签代替空格
df_18.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# 确认更改
df_18.head(1)

# 确认 2008 和 2018 数据集的列标签相同
df_08.columns == df_18.columns

# 确定所有的列标签都相同，如下所示
(df_08.columns == df_18.columns).all()

# 修改列的取值范围
df['No-show'].replace({'No':0, 'Yes':1}, inplace=True)
df['label'] = df.label.map({'ham':0, 'spam':1})
df['label'] = df[xxx].map({'ham':0, 'spam':1}).astype(int)
```

#### 03.03 确认空置
```
# 检查 2008 年的任何列是否有空值 - 应打印为“假”
df_08.isnull().sum().any()
```

#### 03.04
通常情况下删除行，使用参数axis = 0，删除列的参数axis = 1,通常不会这么做，那样会删除一个变量。
```
df.dropna(axis=0)
```

#### 03.05 丢弃含有缺失值的行
```
# 丢弃两个数据集中有任何空值的行
df_08_r.dropna(inplace = True)

# 删除指定行
df_08_r.drop(row_index)

# 检查每个特征值是否含有空值
df_08_r.isnull().sum()         

# 检查所有特征是否含有空值, 如果无, 则返回False
df_08_r.isnull().sum().any()

# 显示数据框的简明摘要，
# 包括每列非空值的数量
# 统计数据缺失
df.info() # 和df_08_r.isnull().sum()效果一样

# 查看该列(cyl)有哪些值
df['cyl'].unique()
```

#### 03.06 从列中提取整型
```
# 检查 2008 cyl 列的值数量
df_08['cyl'].value_counts()
Out[2]:
(6 cyl)     409
(4 cyl)     283
(8 cyl)     199
(5 cyl)      48
(12 cyl)     30
(10 cyl)     14
(2 cyl)       2
(16 cyl)      1

# 从 2008 cyl 列的字符串中提取整数
df_08['cyl'] = df_08['cyl'].str.extract('(\d+)').astype(int)

# or直接将 2018 cyl 列转换成整数，此时不含‘()’
df_18['cyl'] = df_18['cyl'].astype(int)
```

#### 03.07 查看fuel列包含'/'的行
```
hb_18 = df_18[df_18['fuel'].str.contains('/')]
```
#### 03.08 合并类型
Pandas Merges，这是合并数据帧的另一种方式。这类似于数据库风格的 "join"。如果你熟悉 SQL，这个与 SQL 的比较 可以帮助你将它们联系起来。
这里介绍了 Pandas 中的四种合并类型。下文中，"键"指我们将进行连接的两个数据帧中的共同列。
- 内联 使用两个帧中的键的交集。
- 外联 使用两个帧的键的并集。
- 左联 仅使用来自左帧的键。
- 友联 仅使用来自右帧的键。


```
# 修改列名1
df_08.rename(columns=lambda x: x[:10] + "_2008", inplace=True) 
# if colnames are too long, we can limit the characters!

# 修改列名2
new_columns = {
    'PatientId':'patient_id',
    'AppointmentID':'appointment_id',
    'Gender':'gender',
    'ScheduledDay':'scheduled_day',
    'AppointmentDay':'appointment_day',
    'Age':'age',
    'Neighbourhood':'neighbourhood',
    'Scholarship':'scholarship',
    'Hipertension':'hipertension',
    'Diabetes':'diabetes',
    'Alcoholism':'alcoholism',
    'Handcap':'handcap',
    'SMS_received':'sms_received',
    'No-show':'no_show'
}
df.rename(columns=new_columns, inplace=True)
df.columns

df_combined = df_08.merge(df_18, left_on='model_2008', right_on='model', how='inner')
# df_combined = pd.merge(df_08, df_18, left_on='model_2008', right_on='model', how='inner')
df_combined.to_csv('combined_dataset.csv', index=False)

comb_df = pd.read_csv('combined_dataset.csv')
comb_df.shape #(928, 26)

model_mpg_08 = comb_df.groupby('model')['cmb_mpg_2008'].mean(); 
model_mpg_18 = comb_df.groupby('model')['cmb_mpg'].mean(); 

mpg_change = model_mpg_18 - model_mpg_08
mpg_change.describe()

comb_df.query('cmb_mpg - cmb_mpg_2008 > 16')['model']
```
#### 03.09 other
###### 03.09.01 dateframe in 
```
f_08 = df_08.query('fuel in ["CNG", "ethanol"]')['model'].nunique() //值类型的数量
f_08 
```

###### 03.09.02 apply
```
genres = []
df_15['genres'].apply(lambda x: [genres.append(y) for y in x.split('|')])
genres_frq = pd.Series(genres).value_counts()
genres_frq.plot(kind="bar", title="the most popular movie genre of 2015",color='green', legend=True, label="genres freq");
```

###### 03.09.03 sort_values
```
df_2015 = df.query("release_year=='2015'").sort_values("popularity", ascending=False)
df_2015
```

###### 03.09.04 获取满足特定条件列的行
```
result = df[(df['neighbourhood'] == "UNIVERSITÁRIO") & (df['no_show'] == 1)]['no_show'].value_counts()
result
out-> 1    32
result.get_value(label=1)
out-> 32
```
###### 03.09.05 dataframe sum
获取该行的值

|是否幸存	 |幸存者	|遇难者  |
| --------    | -----:   | :----: |
|性别	     |	        |        |
|女性	     |233	    |81      |
|男性	     |109	    |468     |

```
df.sum(axis=0)
out:
性别
女性    314
男性    577
dtype: int64
```
###### 03.09.06 insert 插入列
```
df_t.insert(2, '幸存者比例', groups_data_df_t['幸存者']/groups_data_df.sum(axis=0)) # 插入到第3列
```

###### 03.09.07 统计分析
```
groups_data = passengers_df.groupby(['船舱等级', '是否幸存']).size() #获取花括号形式的数据
groups_data_uns = groups_data.unstack() # 转换为表格
# groups_data_uns.T # 转置
groups_data_uns


groups_data_uns.insert(2, '幸存者比例', groups_data_uns['幸存者']/groups_data_uns.sum(axis=1))# 插入统计的列 幸存者比例
groups_data_uns.columns.name = None  # 为了数据框数据显示明确，这里设为空


# 可视化 x 轴坐标标签方向
var_rot = 'horizontal'
groups_data_uns[['幸存者','遇难者']].plot(kind='bar', stacked=True, alpha=0.6)
plt.ylabel("number")
# 幸存者比例折线图
# groups_data_df_t['幸存者比例'].plot(secondary_y=True, kind='line', rot=var_rot)
groups_data_uns['幸存者比例'].plot(secondary_y=True, kind='line', style='gd-', rot=var_rot)
plt.ylabel("scale")

plt.show()
```
[泰坦尼克号幸存者统计](https://github.com/tynbl/udacity_mls11_lingjian/blob/master/p4/investigate_titanic_data.ipynb "泰坦尼克号幸存者统计")


### 05 基本的SQL
----------

**primary_proc** ：Parch&Posey数据库中的列名称
**web_events** ：Parch&Posey数据库中的表名称
**Database**共享存储在计算机中的链接数据的表的集合
**ERD**显示数据如何在数据库中构造的图表
**SQL**允许我们访问存储在数据库中的数据语言


**表和变量名中不需要空格**
通常在列名中使用下划线，避免使用空格。 在 SQL 中使用空格有点麻烦。 在 Postgres 中，如果列或表名称中有空格，就需要使用双引号括住这些列/表名称（例如：FROM **\"Table Name\"**，而不是 FROM table_name）。在其他环境中，可能会使用方括号（例如：FROM [Table Name]）。

SQL 不区分大小写
```
SeLeCt AcCoUnt_id FrOm oRdErS
```

##### 05.01 limit 加快查询速度
```
SELECT * FROM orders LIMIT 10;
```
##### 05.02 ORDER BY
`ORDER BY` 语句中的列之后添加 `DESC`，然后按降序排序，因为默认是按升序排序的
```
select id, occurred_at, total_amt_usd from orders order by total_amt_usd desc limit 5
# 多个order by
SELECT *
FROM orders
ORDER BY occurred_at, total_amt_usd
LIMIT 10;
```
你会注意到我们在使用这些 WHERE 语句时，不需要 ORDER BY，除非要实际整理数据。不必对数据进行排序，仍可继续执行条件。

##### 05.03 where
```
SELECT name, website, primary_poc FROM accounts WHERE name = 'Exxon Mobil';
```
##### 05.04  as 派生列
查找每个订单海报纸的收入百分比
```
select id,account_id, poster_amt_usd/(gloss_amt_usd + standard_amt_usd + total_amt_usd) as post_per from orders;
```

##### 05.05  LIKE
可用于进行类似于使用 WHERE 和 = 的运算，但是这用于你可能 不 知道自己想准确查找哪些内容的情况。
```
#名字以C开头的账户
SELECT name FROM accounts WHERE name LIKE 'C%';
#名字中间包含one的账户
SELECT name FROM accounts WHERE name LIKE '%one%';
#名字以s结尾的账户
SELECT name FROM accounts WHERE name LIKE '%s';
```
##### 05.06  IN 
用于执行类似于使用 WHERE 和 = 的运算，但用于多个条件的情况。
```
# 查找'Walmart','Target','Nordstrom'的name,primary_poc,sales_rep_id
select name,primary_poc,sales_rep_id  from accounts where name in ( 'Walmart','Target','Nordstrom');
```
##### 05.07  NOT
这与 IN 和 LIKE 一起使用，用于选择 NOT LIKE 或 NOT IN 某个条件的所有行。
```
# 名字不是'Walmart','Target','Nordstrom'
select name,primary_poc,sales_rep_id  from accounts where name not in ('Walmart','Target','Nordstrom');
# 不以C开头的公司
select * from accounts where name not like 'C%';
```
##### 05.08  AND & BETWEEN
可用于组合所有组合条件必须为真的运算。
```
# 使用 web_events 表查找通过 organic 或 adwords 联系，
# 并在 2016 年的任何时间开通帐户的个人全部信息，并按照从最新到最旧的顺序排列.
select * from web_events where channel in ('organic','adwords') and occurred_at BETWEEN '2016-01-01' AND '2017-01-01' order by occurred_at desc;
```
##### 05.09 OR
可用于组合至少一个组合条件必须为真的运算。
```
# gloss_qty 或 poster_qty 大于 4000
select * from orders where gloss_qty > 4000 or poster_qty > 4000;
# 标准数量 (standard_qty)为零，光泽度 (gloss_qty) 或海报数量 (poster_qty)超过 1000
select * from orders where standard_qty = 0 and (gloss_qty > 1000 or poster_qty > 1000);

#
SELECT *
FROM accounts
WHERE (name LIKE 'C%' OR name LIKE 'W%')
           AND ((primary_poc LIKE '%ana%' OR primary_poc LIKE '%Ana%') 
           AND primary_poc NOT LIKE '%eana%');
```

### 06 SQL Join
- 使得数据更易组织
- 多表结构可以保证更快的查询
查询速度决定于你需要数据库读取的数据量
以及需要进行的计算数量和类型

#### 06.01 ERD 实体关系图 
![erd_example](https://s3.cn-north-1.amazonaws.com.cn/u-img/dac4dd37-f8c7-4249-9c64-cb9716022ed3)

`PK`表示主键。每个表格都存在主键，它是每行的值都唯一的列。
外键 (`FK`):外键是另一个表格中的主键。

#### 06.02 inner join
下面列出了您可以使用的 `JOIN` 类型，以及它们之间的差异。
`JOIN`: 如果表中有至少一个匹配，则返回行
`LEFT JOIN`: 即使右表中没有匹配，也从左表返回所有的行
`RIGHT JOIN`: 即使左表中没有匹配，也从右表返回所有的行
`FULL JOIN`: 只要其中一个表中存在匹配，就返回行

`OUTER JOIN`
最后一种连接类型是外连接，它将返回**内连接的结果，以及被连接的表格中没有匹配的行**。
这种连接返回的是与两个表格中的某个表格不匹配的行，**完整的外连接用例非常罕见**。

虽然 SQL 没有强制要求 ON 语句必须使**主键等于外键**，但是我们编写的语句基本都是这种情况（当然也有一些极端例外情况）

#### 06.03 别名
我们可以直接在列名称（在 SELECT 中）或表格名称（在 FROM 或 JOIN 中）后面写上别名，方法是在要**设定别名的列或表格后面直接写上别名**。这样可以创建清晰的列名称，虽然计算是用来创建列的，通过使用表格名称别名，代码更高效。’

#### 06.04 NULL
任何没有数据的单元格都是 NULL
```
select * from where xx is NULL
```
#### 06.05 实例
```
# 两张表连接
SELECT Persons.LastName, Persons.FirstName, Orders.OrderNo
FROM Persons
INNER JOIN Orders
ON Persons.Id_P = Orders.Id_P
ORDER BY Persons.LastName;

# 两张表连接
select orders.standard_qty,orders.gloss_qty,orders.poster_qty,accounts.website,accounts.primary_poc
FROM orders
JOIN accounts
ON orders.account_id = accounts.id;

# 两张表连接 + 别名
select a.name,a.primary_poc,w.occurred_at,w.channel
from accounts a
join web_events w
on a.id = w.account_id
where a.name = 'Walmart';

# 多张表连接1
SELECT web_events.channel, accounts.name, orders.total
FROM web_events
JOIN accounts
ON web_events.account_id = accounts.id
JOIN orders
ON accounts.id = orders.account_id;

# 多张表连接2
select region.name region, sales_reps.name sales_reps, accounts.name accounts
from sales_reps
join region
on sales_reps.region_id = region.id
join accounts
on sales_reps.id = accounts.sales_rep_id
order by accounts.name;

# 多张表连接3 + 别名
select r.name region, a.name accounts,
o.total_amt_usd/(0.01 + o.standard_amt_usd + o.gloss_amt_usd + o.poster_amt_usd) per_price
from sales_reps s
join region r
on s.region_id = r.id
join accounts a
on s.id = a.sales_rep_id
join orders o
on o.account_id = a.id;

# 多张表连接4 + 别名 + DISTINCT
SELECT DISTINCT w.id, w.occurred_at, a.name, o.total, o.total_amt_usd
FROM accounts a
JOIN orders o
ON o.account_id = a.id
JOIN web_events w
ON a.id = w.account_id
WHERE w.occurred_at BETWEEN '01-01-2015' AND '01-01-2016'
ORDER BY w.occurred_at DESC;
```

#### 06.06 left join & right join
```
# 如果left表中的行在right中没有匹配到，则返回left表中的行
select xxx from left table 
left join right table
on xx.xx = xx.xx

# 如果right表中的行在left中没有匹配到，则返回right表中的行
select xxx from left table 
right join right table
on xx.xx = xx.xx
```
#### 06.06.02 以下连接等价
```
LEFT OUTER JOIN == LEFT JOIN
RIGHT OUTER JOIN == RIGHT JOIN
```

#### 06.07 join 和 过滤
```
select xxx from left table 
right join right table
on xx.xx = xx.xx 
and # join前筛选，减少行
# logic in the on clause reduces the rows before combining the tables

where # join查询之后在筛选，减少行
# logic in the where clause occurs after the join occurs
```

#### 06.08 总结
`JOIN`
在这节课，你学习了如何使用 `JOIN` 组合多个表格的数据。
`JOIN` 		- 一种 `INNER JOIN`，仅获取在两个表格中都存在的数据。
`LEFT JOIN` 	- 用于获取 `FROM` 中的表格中的所有行，即使它们不存在于 `JOIN` 语句中。
`RIGHT JOIN` 	- 用于获取 `JOIN` 中的表格中的所有行，即使它们不存在于 `FROM` 语句中。

还有几个没有讲解的高级 `JOIN`，它们仅适用于非常特定的情况。`UNION` 和 `UNION ALL`、`CROSS` `JOIN` 和比较难的 `SELF JOIN`。这些内容比较深奥，这门课程不会再过多介绍，但是有必要知道这些连接方法是存在的，它们在特殊情况下比较实用。


### 07 SQL聚合
#### 07.1 count
数值 + 文本
```
COUNT 不会考虑具有 NULL 值的行。因此，可以用来快速判断哪些行缺少数据。
```
#### 07.2 SUM MIN MAX AVG
与 COUNT 不同，你只能针对数字列使用SUM。 但是，SUM 将**忽略`NULL`值**,null值被当作0处理，其他聚合函数也是这样.
**聚合函数只能垂直聚合**，即聚合列的值。如果你想对行进行计算，可以使用[简单算术表达式](https://community.modeanalytics.com/sql/tutorial/sql-operators/#arithmetic-in-sql "简单算术表达式")
```
select min(gloss_qty) from orders where gloss_qty > 10;
select max(gloss_qty) from orders where gloss_qty > 10;
select avg(gloss_qty) from orders where gloss_qty > 10;
```
- **MIN** 和 **MAX** 聚合函数也会忽略 NULL 值;
- 从功能上来说，**MIN** 和 **MAX** 与 **COUNT** 相似，它们都可以用在非数字列上。MIN 将返回最小的数字、最早的日期或按字母表排序的最之前的非数字值，具体取决于列类型。MAX 则正好相反，返回的是最大的数字、最近的日期，或与“Z”最接近（按字母表顺序排列）的非数字值。
- **avg** 返回的是数据的平均值，即列中所有的值之和除以列中值的数量（**null值不会记入分子分母**，如果要，则需要sum/count）。该聚合函数同样会忽略分子和分母中的 NULL 值。

#### 07.3 group by
主要知识点包括：
1. GROUP BY 可以用来在数据子集中聚合数据。例如，不同客户、不同区域或不同销售代表分组。
1. SELECT 语句中的任何一列如果不在聚合函数中，则必须在 GROUP BY 条件中。
1. GROUP BY 始终在 WHERE 和 ORDER BY 之间。
1. ORDER BY 有点像电子表格软件中的 SORT。

```
# 每个客户的最小订单，并按照升序排列
SELECT a.name, MIN(total_amt_usd) smallest_order
FROM accounts a
JOIN orders o
ON a.id = o.account_id
GROUP BY a.name
ORDER BY smallest_order;
```
从此例可以看出就 **Min Max Sum Count等函数优先级**小于`Group By`, 大于`Order by`;



#### 07.4 group by 多列分组
主要知识点：

你可以同时按照多列分组，正如此处所显示的那样。这样经常可以在大量不同的细分中更好地获得聚合结果。
ORDER BY 条件中列出的列顺序有区别。你是从**左到右让列排序**。

##### 专家提示
- GROUP BY 条件中的**列名称顺序并不重要**，结果还是一样的。如果运行相同的查询并颠倒 GROUP BY 条件中列名称的顺序，可以看到结果是一样的。
- 和 ORDER BY 一样，你可以在 GROUP BY 条件中用数字替换列名称。仅当你对大量的列分组时，或者其他原因导致 GROUP BY 条件中的文字过长时，才建议这么做。
- 提醒下，任何不在聚合函数中的列必须显示 GROUP BY 语句。如果忘记了，可能会遇到错误。但是，即使查询可行，你也可能不会喜欢最后的结果！


#### 07.5 DISTINCT
`DISTINCT` 看做仅返回特定列的唯一值的函数。

##### 专家提示
需要注意的是，在使用`DISTINCT`时，尤其是在聚合函数中使用时，会让查询速度有所减慢。
```
检查是否有任何客户与多个区域相关联
select distinct a.id, a.name, r.id, r.name
from accounts a
join sales_reps s
on s.id = a.sales_rep_id
join region r
on r.id = s.region_id;

select distinct id from accounts;
```

####  07.6 HAVING
HAVING 是过滤被聚合的查询的 the “整洁”方式，但是通常采用子查询的方式来实现。本质上，只要你想对通过聚合创建的查询中的元素执行 WHERE 条件，就需要使用 HAVING。
**having 必须在group by之后，并且和聚合函数一起使用**

- WHERE 子集根据逻辑条件对**返回**的数据进行筛选。
- WHERE 出现在 FROM，JOIN 和 ON 条件之后，但是在 GROUP BY 之前。
- HAVING 出现在 GROUP BY 条件之后，但是在 **ORDER BY 条件之前。
- HAVING 和 WHERE 相似，但是它**适合涉及聚合**的逻辑语句。

#### 07.7 子查询
```
SELECT COUNT(*) num_reps_above5
FROM(SELECT s.id, s.name, COUNT(*) num_accounts
     FROM accounts a
     JOIN sales_reps s
     ON s.id = a.sales_rep_id
     GROUP BY s.id, s.name
     HAVING COUNT(*) > 5
     ORDER BY num_accounts) AS Table1;
```

#### 07.8 DATE函数

首先，我们要介绍的日期函数是 DATE_TRUNC。

[DATE_TRUNC](https://blog.modeanalytics.com/date-trunc-sql-timestamp-function-count-on/ "DATE_TRUNC") 使你能够**将日期截取到日期时间列的特定部分**。常见的截取依据包括日期、月份 和 年份。

DATE_PART 可以用来**获取日期的特定部分**，但是注意获取 month 或 dow 意味着无法让年份按顺序排列。而是按照特定的部分分组，无论它们属于哪个年份。
[DateTime](https://www.postgresql.org/docs/9.1/static/functions-datetime.html)

Parch & Posey 在哪一年的总销售额最高？数据集中的所有年份保持均匀分布吗？
```
select DATE_TRUNC('year', occurred_at) AS year, sum(total_amt_usd) total_year
from orders
group by 1
order by 2 desc;
```
Parch & Posey 在哪一个月的总销售额最高？数据集中的所有月份保持均匀分布吗？
```
select DATE_TRUNC('month', occurred_at) AS month, sum(total_amt_usd) total_year
from orders
group by 1
order by 2 desc;
```

Parch & Posey 在哪一年的总订单量最多？数据集中的所有年份保持均匀分布吗？
```
select DATE_TRUNC('year', occurred_at) AS year, count(total_amt_usd) total_year
from orders
group by 1
order by 2 desc;
```
Parch & Posey 在哪一个月的总订单量最多？数据集中的所有月份保均匀分布吗？
```
select DATE_TRUNC('month', occurred_at) AS month, count(total_amt_usd) total_year
from orders
group by 1
order by 2 desc;
```
Walmart 在哪一年的哪一个月在铜版纸上的消费最多？
```
select DATE_TRUNC('month', occurred_at) AS month, count(total_amt_usd) total_year
from orders
group by 1
order by 2 desc
limit 1;
```

#### 07.9 case 专家提示
- CASE 语句始终位于 SELECT 条件中。
- CASE 必须包含以下几个部分：WHEN、THEN 和 END。
- ELSE 是可选组成部分，用来包含不符合上述任一 CASE 条件的情况。

你可以在 WHEN 和 THEN 之间使用任何条件运算符编写任何条件语句（例如 WHERE），包括使用 AND 和 OR 连接多个条件语句。

你可以再次包含多个 WHEN 语句以及 ELSE 语句，以便处理任何未处理的条件。

```
SELECT account_id,
	   CASE WHEN standard_qty = 0 OR standard_qty IS NULL THEN 0
			WHEN xxx THEN xxx
            ELSE standard_amt_usd/standard_qty END AS unit_price
FROM orders
LIMIT 10;
```
##### 练习
1. 我们想要根据相关的消费量了解三组不同的客户。最高的一组是终身价值（所有订单的总销售额）大于 200,000 美元的客户。第二组是在 200,000 到 100,000 美元之间的客户。最低的一组是低于 under 100,000 美元的客户。请提供一个表格，其中包含与每个客户相关的级别。你应该提供客户的名称、所有订单的总销售额和级别。消费最高的客户列在最上面。
```
select a.name, sum(total_amt_usd) total_spend,
	   CASE WHEN sum(total_amt_usd) > 200000 THEN 'A'
			WHEN sum(total_amt_usd) between 100000 and 200000 THEN 'B'
            ELSE 'C'END AS level
from accounts a
join orders o
on a.id = o.account_id
group by a.name
order by total_spend desc;
```

2. 现在我们想要执行和第一个问题相似的计算过程，但是我们想要获取在 2016 年和 2017 年客户的总消费数额。级别和上一个问题保持一样。消费最高的客户列在最上面。
```
select a.name, sum(total_amt_usd) total_spend,
	   CASE WHEN sum(total_amt_usd) > 200000 THEN 'A'
			WHEN sum(total_amt_usd) between 100000 and 200000 THEN 'B'
            ELSE 'C'END AS level
from accounts a
join orders o
on a.id = o.account_id
where occurred_at between '2016-01-01' and '2018-01-01'
group by a.name
order by total_spend desc;
```
3. 我们想要找出绩效最高的销售代表，也就是有超过 200 个订单的销售代表。创建一个包含以下列的表格：销售代表名称、订单总量和标为 top 或 not 的列（取决于是否拥有超过 200 个订单）。销售量最高的销售代表列在最上面。
```
select s.name, count(*) as all_orders_num,
	   CASE WHEN count(*) > 200 THEN 'top'
            ELSE 'not' END AS youxiu
from accounts a
join orders o
on a.id = o.account_id
join sales_reps s
on s.id = a.sales_rep_id
group by s.name
order by all_orders_num desc;
```
 
4. 之前的问题没有考虑中间水平的销售代表或销售额。管理层决定也要看看这些数据。我们想要找出绩效很高的销售代表，也就是有超过 200 个订单或总销售额超过 750000 美元的销售代表。中间级别是指有超过 150 个订单或销售额超过 500000 美元的销售代表。创建一个包含以下列的表格：销售代表名称、总订单量、所有订单的总销售额，以及标为 top、middle 或 low 的列（取决于上述条件）。在最终表格中将销售额最高的销售代表列在最上面。根据上述标准，你可能会见到几个表现很差的销售代表！
```
select s.name, sum(total_amt_usd) as total_xs, count(*) as all_orders_num,
	   CASE WHEN count(*) > 200 or sum(total_amt_usd) > 750000 THEN 'top'
       		WHEN count(*) > 150 or sum(total_amt_usd) > 500000 THEN 'mid'
            ELSE 'low' END AS youxiu
from accounts a
join orders o
on a.id = o.account_id
join sales_reps s
on s.id = a.sales_rep_id
group by s.name
order by total_xs desc;
```

### 08 SQL子查询和临时表格
- 子查询
- 表格表达式
- 持久衍生表格

子查询和表格表达式都是用来通过查询创建一个表格，然后再编写一个查询来与这个新创建的表格进行互动。有时候，你要回答的问题无法通过直接处理数据库中的现有表格获得答案。

但是，如果我们能通过现有的表格创建新的表格，我们就能查询这些新的表格，并回答我们的问题。这节课的查询就可以实现这一目的。

##### 例：每个渠道平均每个日期发生的交易数
```
select channel, avg(a_num) from
(
select DATE_TRUNC('day', occurred_at) AS day, channel, count(*) a_num
from web_events
group by channel, day
order by 1
) channel_day
group by channel
order by 2
```

#### 08.01 子查询格式
```
SELECT *
FROM (SELECT DATE_TRUNC('day',occurred_at) AS day, # 从哪里查询
channel, COUNT(*) as events
FROM web_events
GROUP BY 1,2            #group by & order by同级别缩进
ORDER BY 3 DESC) sub
GROUP BY channel
ORDER BY 2 DESC;
```

##### 专家提示
注意，在条件语句中编写**子查询时，不能包含别名**。这是因为该子查询会被当做单个值（或者对于 IN 情况是一组值），而不是一个表格。

同时注意，这里的查询对应的是单个值。如果我们返回了**整个列**，则需要使用 **IN** 来执行逻辑参数。如果我们要返回**整个表**格，则必须为该表格使用**别名**，并对整个表格执行其他逻辑。

##### 子查询
###### 1. 提供每个区域拥有最高销售额 (`total_amt_usd`) 的销售代表的姓名。
```
# t1和stub1是相同的表
# 整体结构
############################################
#	select xx
#	from t1
#	join (select max_sale_number from stub1) t2
#	on t1.xx = t2.xx and ..
############################################
select t1.sale_name, t1.per_man_r_total_amt
from (select r.name region_name, s.name sale_name, sum(total_amt_usd) per_man_r_total_amt
from sales_reps s
join region r
on r.id = s.region_id
join accounts a
on a.sales_rep_id = s.id
join orders o
on o.account_id = a.id
group by r.name, s.name
order by per_man_r_total_amt desc) t1 # 每个区域每个人的销售额降序排列
join (
select region_name, max(per_man_r_total_amt) max_man_r_total_amt
from (select r.name region_name, s.name sale_name, sum(total_amt_usd) per_man_r_total_amt
from sales_reps s
join region r
on r.id = s.region_id
join accounts a
on a.sales_rep_id = s.id
join orders o
on o.account_id = a.id
group by r.name, s.name
order by per_man_r_total_amt desc) stub1 # 每个区域每个人的销售额降序排列
group by region_name # 每个区域中单个人的最大销售额
) t2
on t1.region_name = t2.region_name and t1.per_man_r_total_amt = t2.max_man_r_total_amt # t1 t2连接起来可以得到在某区域最大销售额的人的姓名和销售额
```

###### 公共表达式版本
```
with t1 as (select r.name region_name, s.name sale_name, sum(total_amt_usd) per_man_r_total_amt
from sales_reps s
join region r
on r.id = s.region_id
join accounts a
on a.sales_rep_id = s.id
join orders o
on o.account_id = a.id
group by r.name, s.name
order by per_man_r_total_amt desc),

t2 as (
select region_name, max(per_man_r_total_amt) max_man_r_total_amt
from t1
group by region_name
)

select t1.sale_name, t1.per_man_r_total_amt
from t1
join t2
on t1.region_name = t2.region_name and t1.per_man_r_total_amt = t2.max_man_r_total_amt
```

###### 2. 对于具有最高销售额 (total_amt_usd) 的区域，总共下了多少个订单？
```
SELECT r.name region_name, count(*) total_orders, SUM(o.total_amt_usd) total_amt
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY r.name
having SUM(o.total_amt_usd) = (# 找出最大的销售额的区域
SELECT MAX(total_amt) max_total_amt
FROM (SELECT r.name region_name, SUM(o.total_amt_usd) total_amt # 每个区域的销售额
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY r.name) t1
);
```
###### 3. 对于购买标准纸张数量 (standard_qty) 最多的客户（在作为客户的整个时期内），有多少客户的购买总数依然更多？
```
SELECT a.name accounts_name, sum(o.total) account_total
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY accounts_name
having sum(o.total) > (
select account_total
from (SELECT a.name accounts_name, sum(o.standard_qty) account_standard_qty, sum(o.total) account_total
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY accounts_name
order by account_standard_qty desc
limit 1
) t2
);
```
###### 4. 对于（在作为客户的整个时期内）总消费 (total_amt_usd) 最多的客户，他们在每个渠道上有多少 web_events？
```
select w.channel, count(*) account_channel_num
from web_events w
join accounts a
on w.account_id = a.id and a.name = (select accounts_name
from (
# 消费总额最多的客户
SELECT a.name accounts_name, sum(o.total_amt_usd) account_total_amt_usd
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY accounts_name
order by account_total_amt_usd desc
limit 1) t1
)
group by w.channel
order by account_channel_num desc
```

##### 其他例子
###### 5. 对于总消费前十名的客户，他们的平均终身消费 (total_amt_usd) 是多少?

```
# 公共表达式版本
with top_t as (select a.name, sum(o.total_amt_usd) total_spend
from accounts a
join orders o
on o.account_id = a.id
group by a.name
order by total_spend desc
limit 10)

select avg(top_t.total_spend)
from top_t
```
###### 6. 比所有客户的平均消费高的企业平均终身消费 (total_amt_usd) 是多少？
###### 公共表达式版本
```
with t_avg as (SELECT avg(o.total_amt_usd) avg_spent
FROM orders o
JOIN accounts a
ON a.id = o.account_id),

t2 as (select o.account_id, avg(o.total_amt_usd) avg_t_spend
FROM orders o
group by o.account_id
having avg(o.total_amt_usd) > (select * from t_avg))

select avg(avg_t_spend) from t2;
```

#### 08.02 With
WITH 语句经常称为公用表表达式（简称 CTE）。虽然这些表达式和子查询的目的完全一样，但是实际更常用，因为对未来的读者来说，更容易看懂其中的逻辑。
优点：完全独立执行查询，并写入数据库，提高查询速度

##### 单个with语句
```
with [table-alias] as (
	此处相当于子查询
)

select xx1, xx2
from [table-alias]
group by xx1,xx2
order by xx2 desc
```
##### 多个with语句
```
WITH table1 AS (
SELECT *
FROM web_events), # 此处逗号分隔

table2 AS (
SELECT *
FROM accounts)


SELECT *
FROM table1
JOIN table2
ON table1.account_id = table2.id;
```
- 创建多个表格时，需要在每个表格后面加一个逗号，但是在引向最终查询的最后一个表格后面不需添加。
- 新表格名称始终使用 table_name AS 设置别名，后面是位于小括号中的查询。


### 09 SQL数据清理
```
1. 清理和重新整理混乱的数据。
2. 将列转换为不同的数据类型。
3. 处理 NULL 的技巧。
```

#### 09.01 LEFT & RIGHT
- LEFT 从起点（或左侧）开始，从特定列中的每行获取一定数量的字符。正如此处看到的，你可以使用 LEFT(phone_number, 3) 获取电话号码中的前三位。
- RIGHT 从末尾（或右侧）开始，从特定列中的每行获取一定数量的字符。正如此处看到的，你可以使用 RIGHT(phone_number, 8) 获取电话号码的最后 8 位。
- LENGTH 提供了特定列每行的字符数。这里，我们可以使用 LENGTH(phone_number) 得出每个电话号码的长度。
###### 练习
##### 09.01.01 在 accounts 表格中，有一个列存储的是每个公司的网站。最后三个数字表示他们使用的是什么类型的网址。
此处给出了扩展（和价格）列表。请获取这些扩展并得出 accounts 表格中每个网址类型的存在数量。
```
select right(website, 3) as web_type, count(*) web_num
from accounts
group by web_type
order by web_num desc
```
##### 09.01.02 对于公司名称（甚至名称的第一个字母）的作用存在颇多争议(https://www.quora.com/Does-a-companys-name-matter)。
请从 accounts 表格中获取每个公司名称的第一个字母，看看以每个字母（数字）开头的公司名称分布情况。
```
select left(upper(name), 1) prefix, count(*) pre_number
from accounts
group by prefix
order by pre_number desc
```
##### 09.01.03 使用 accounts 表格和 CASE 语句创建两个群组：一个是以数字开头的公司名称群组，另一个是以字母开头的公司名称群组。以字母开头的公司名称所占的比例是多少？
```
元音是指 a、e、i、o 和 u。有多少比例的公司名称以元音开头，以其他音节开头的公司名称百分比是多少？
with t as (SELECT name, CASE WHEN LEFT(UPPER(name), 1) IN ('A','E','I','O','U') 
                       THEN 1 ELSE 0 END AS num, 
         CASE WHEN LEFT(UPPER(name), 1) IN ('A','E','I','O','U') 
                       THEN 0 ELSE 1 END AS letter
      FROM accounts)
select sum(num) a, sum(letter) b from t;
```

#### 09.02 POSITION STRPOS UPPER LOWER
1. POSITION 获取字符和列，并提供该字符在每行的索引。**第一个位置的索引在 SQL 中是 1**。如果你之前学习了其他编程语言，就会发现很多语言的索引是从 0 开始。这里，你发现可以使用 POSITION(',' IN city_state) 获取逗号的索引。
```
select website, position('.' in website) pos
from accounts
```
1. **STRPOS** 和 **POSITION** 提供的结果相同，但是语法不太一样，如下所示：STRPOS(city_state, ‘,’)。
1. 注意，POSITION 和 STRPOS 都**区分大小写**，因此查找 A 的位置与查找 a 的结果不同。
1. 因此，如果你想获取某个字母的索引，但是不区分大小写，则需要使用 LOWER 或 UPPER 让所有字符变成小写或大写。

###### 练习
##### 09.02.01 使用 accounts 表格创建一个名字和姓氏列，用于存储 primary_poc 的名字和姓氏。
```
select primary_poc , left(primary_poc, position(' ' in primary_poc) - 1) first_name, right(primary_poc, length(primary_poc) - position(' ' in primary_poc)) last_name
from accounts
```
##### 09.02.02 现在创建一个包含 sales_rep 表格中每个销售代表姓名的列，同样，需要提供名字和姓氏列。

#### 09.03 CONCAT
- CONCAT
- Piping ||

这两个工具都能将不同行的列组合到一起。在此视频中，你学习了如何将存储在不同列中的名字和姓氏组合到一起，形成全名：`CONCAT(first_name, ' ', last_name)`，或者使用双竖线：`first_name || ' ' || last_name`。
###### 练习
##### 09.03.01 accounts 表格中的每个客户都想为每个 primary_poc 创建一个电子邮箱。邮箱应该是 primary_poc 的名字.primary_poc的姓氏@公司名称.com。
```
select left(primary_poc, position(' ' in primary_poc) - 1) || '.' || right(primary_poc, length(primary_poc) - position(' ' in primary_poc)) || '@' || name || '.com' as email
from accounts
```
##### 09.03.02 你可能注意到了，在上一个答案中，有些公司名称存在空格，肯定不适合作为邮箱地址。看看你能否通过删掉客户名称中的所有空格来创建合适的邮箱地址，否则你的答案就和问题 1. 的一样。此处是一些实用的文档。
```
select left(primary_poc, position(' ' in primary_poc) - 1) || '.' || right(primary_poc, length(primary_poc) - position(' ' in primary_poc)) || '@' || replace(name, ' ','') || '.com' as email
from accounts
```
##### 09.03.03 我们还需要创建初始密码，在用户第一次登录时将更改。初始密码将是 primary_poc 的名字的第一个字母（小写），然后依次是名字的最后一个字母（小写）、姓氏的第一个字母（小写）、姓氏的最后一个字母（小写）、名字的字母数量、姓氏的字母数量，然后是合作的公司名称（全大写，没有空格）

#### 09.04 TO_DATE CAST
```
1. TO_DATE
2. CAST
3. 使用 :: 进行转型
```
`DATE_PART('month', TO_DATE(month, 'month'))`将月份名称改成了与该月相关的数字。
然后，你可以使用 `CAST` 将字符串改为日期。`CAST` 实际上可以用来更改各种列类型。经常，你会像视频中一样，使用 `CAST(date_column AS DATE)` 将字符串改成日期。但是，你可能还会对列的数据类型做出其他更改。你可以在此处看到其他例子。
在此示例中，除了 `CAST(date_column AS DATE)` 之外，你可以使用 `date_column::DATE`。
###### 练习
##### 09.04.01 规范日期格式

- `substr(string, from , length)` from:起始位置， length:要截取的长度
- `date_column::DATE`

```
select (right(substr(date,1,10),4) ||  left(substr(date,1,10),2) || substr(substr(date,1,10),4,2))::date as ymd from sf_crime_data
limit 10;

SELECT date orig_date, (SUBSTR(date, 7, 4) || '-' || LEFT(date, 2) || '-' || SUBSTR(date, 4, 2))::DATE new_date
FROM sf_crime_data
limit 10;
```
#### 09.04 COALESCE
`COALESCE` 返回的是每行的**第一个非 NULL 值**。因此如果在此示例中，行中的值是 NULL，上述解决方案使用了 no_poc
- count等函数将null当作0处理，将null赋予别的初值，可以计算null值数量

### 10 绘图
#### 10.01 Series
#### 10.01.01 Series
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

fig, axes = plt.subplots(2, 2) # 两行两列
s = pd.Series(np.random.randn(20).cumsum(), index=np.arange(0, 100, 5))
s.plot(kind='bar', ax=axes[0][0])
s.plot(kind='bar', ax=axes[0][1],stacked=True)
s.plot(kind='barh', ax=axes[1][0],grid='True') # 显示网格线
s.plot(kind='line', ax=axes[1][1], color='k')
plt.show()
```

#### 10.01.02 Series.plot方法的函数
```
参数	说明
label	用于图例的标签
ax	    要在其上进行绘制的matplotlib subplot对象。如果没有设置，则使用当前matplotlib subplot
style	将要传给matplotlib的风格字符串(for example: ‘ko–’)
alpha	图表的填充不透明(0-1)
kind	可以是’line’, ‘bar’, ‘barh’, ‘kde’
logy	在Y轴上使用对数标尺
use_index	将对象的索引用作刻度标签
rot	    旋转刻度标签(0-360)
xticks	用作X轴刻度的值
yticks	用作Y轴刻度的值
xlim	X轴的界限
ylim	Y轴的界限
grid	显示轴网格线
```
#### 10.02 DataFrame
DataFrame 是一个表格型的数据结构。它提供有序的列和不同类型的列值
#### 10.02.01 DataFrame
```
from pandas import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#随机生成6行4列
df=DataFrame(np.random.rand(6,4),index=['one','two','three','four','five','six'],columns=['A','B','C','D'])
df.columns.name = 'Genus'
print(df)
df.plot(kind='bar') # 4列为一组，共6组的直方图
plt.show()

# 4条连续的折线图
df = pd.DataFrame(np.random.randn(20, 4).cumsum(0), columns=list('ABCD'), index=np.arange(0, 100, 5))
df.plot()
plt.show()
```
###### 4列为一组，共6组的直方图
![直方图_4.png](https://i.imgur.com/vKt0PPr.png)

###### 4条连续的折线图
![折线图_2.png](https://i.imgur.com/psMrJis.png)


#### 10.02.02 DataFrame的方法参数
```
参数	说明
subplots	将各个DataFrame列绘制到单独的subplot中
sharex	    如果subplots=True，则共用同一个X轴，包括刻度和界限
sharey	    类似于上
figsize	    表示图像大小的元组
title	    表示图像标题的字符串
legend	    添加一个subplot图例(默认为True)
sort_columns以字母表顺序绘制各列，默认使用前列顺序
```

#### 10.02.03 DataFrame 取值
```
import numpy as np
import pandas as pd
from pandas import Sereis, DataFrame

ser = Series(np.arange(3.))

data = DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('wxyz'))

data['w']  #选择表格中的'w'列，使用类字典属性,返回的是Series类型

data.w    #选择表格中的'w'列，使用点属性,返回的是Series类型

data[['w']]  #选择表格中的'w'列，返回的是DataFrame类型

data[['w','z']]  #选择表格中的'w'、'z'列

data[0:2]  #返回第1行到第2行的所有行，前闭后开，包括前不包括后

data[1:2]  #返回第2行，从0计，返回的是单行，通过有前后值的索引形式，
       #如果采用data[1]则报错

data.ix[1:2] #返回第2行的第三种方法，返回的是DataFrame，跟data[1:2]同

data['a':'b']  #利用index值进行切片，返回的是**前闭后闭**的DataFrame, 
        #即末端是包含的  

data.head()  #返回data的前几行数据，默认为前五行，需要前十行则data.head(10)
data.tail()  #返回data的后几行数据，默认为后五行，需要后十行则data.tail(10)

data.iloc[-1]   #选取DataFrame最后一行，返回的是Series
data.iloc[-1:]   #选取DataFrame最后一行，返回的是DataFrame

data.loc['a',['w','x']]   #返回‘a’行'w'、'x'列，这种用于选取行索引列索引已知

data.iat[1,1]   #选取第二行第二列，用于已知行、列位置的选取

# 从数据集中选择三个你希望抽样的数据点的索引
indices = [23,39,211]

# 为选择的样本建立一个DataFrame，并且重新索引
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
```

#### 10.02.04 DataFrame取值例子
**当用已知的行索引时为前闭后闭区间，这点与切片稍有不同**。
```
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

data = DataFrame(np.arange(15).reshape(3,5),index=['one','two','three'],columns=['a','b','c','d','e'])

data
Out[7]: 
        a   b   c   d   e
one     0   1   2   3   4
two     5   6   7   8   9
three  10  11  12  13  14

#对列的操作方法有如下几种

data.icol(0)   #选取第一列
E:\Anaconda2\lib\site-packages\spyder\utils\ipython\start_kernel.py:1: FutureWarning: icol(i) is deprecated. Please use .iloc[:,i]
  # -*- coding: utf-8 -*-
Out[35]: 
one       0
two       5
three    10
Name: a, dtype: int32

data['a']
Out[8]: 
one       0
two       5
three    10
Name: a, dtype: int32

data.a
Out[9]: 
one       0
two       5
three    10
Name: a, dtype: int32

data[['a']]
Out[10]: 
        a
one     0
two     5
three  10

data.ix[:,[0,1,2]]  #不知道列名只知道列的位置时
Out[13]: 
        a   b   c
one     0   1   2
two     5   6   7
three  10  11  12

data.ix[1,[0]]  #选择第2行第1列的值
Out[14]: 
a    5
Name: two, dtype: int32

data.ix[[1,2],[0]]   #选择第2,3行第1列的值
Out[15]: 
        a
two     5
three  10

data.ix[1:3,[0,2]]  #选择第2-4行第1、3列的值
Out[17]: 
        a   c
two     5   7
three  10  12

data.ix[1:2,2:4]  #选择第2-3行，3-5（不包括5）列的值
Out[29]: 
     c  d
two  7  8

data.ix[data.a>5,3]
Out[30]: 
three    13
Name: d, dtype: int32

data.ix[data.b>6,3:4]  #选择'b'列中大于6所在的行中的第4列，有点拗口
Out[31]: 
        d
three  13

data.ix[data.a>5,2:4]  #选择'a'列中大于5所在的行中的第3-5（不包括5）列
Out[32]: 
        c   d
three  12  13

data.ix[data.a>5,[2,2,2]]  #选择'a'列中大于5所在的行中的第2列并重复3次
Out[33]: 
        c   c   c
three  12  12  12

#还可以行数或列数跟行名列名混着用
data.ix[1:3,['a','e']]
Out[24]: 
        a   e
two     5   9
three  10  14

data.ix['one':'two',[2,1]]
Out[25]: 
     c  b
one  2  1
two  7  6

data.ix[['one','three'],[2,2]]
Out[26]: 
        c   c
one     2   2
three  12  12

data.ix['one':'three',['a','c']]
Out[27]: 
        a   c
one     0   2
two     5   7
three  10  12

data.ix[['one','one'],['a','e','d','d','d']]
Out[28]: 
     a  e  d  d  d
one  0  4  3  3  3
one  0  4  3  3  3

#对行的操作有如下几种：
data[1:2]  #（不知道列索引时）选择第2行，不能用data[1]，可以用data.ix[1]
Out[18]: 
     a  b  c  d  e
two  5  6  7  8  9

data.irow(1)   #选取第二行
Out[36]: 
a    5
b    6
c    7
d    8
e    9
Name: two, dtype: int32

data.ix[1]   #选择第2行
Out[20]: 
a    5
b    6
c    7
d    8
e    9
Name: two, dtype: int32


data['one':'two']  #当用已知的行索引时为前闭后闭区间，这点与切片稍有不同。
Out[22]: 
     a  b  c  d  e
one  0  1  2  3  4
two  5  6  7  8  9

data.ix[1:3]  #选择第2到4行，不包括第4行，即前闭后开区间。
Out[23]: 
        a   b   c   d   e
two     5   6   7   8   9
three  10  11  12  13  14

data.ix[-1:]  #取DataFrame中最后一行，返回的是DataFrame类型,**注意**这种取法是有使用条件的，只有当行索引不是数字索引时才可以使用，否则可以选用`data[-1:]`--返回DataFrame类型或`data.irow(-1)`--返回Series类型
Out[11]: 
        a   b   c   d   e
three  10  11  12  13  14

data[-1:]  #跟上面一样，取DataFrame中最后一行，返回的是DataFrame类型
Out[12]: 
        a   b   c   d   e
three  10  11  12  13  14

data.ix[-1] #取DataFrame中最后一行，返回的是Series类型，这个一样，行索引不能是数字时才可以使用
Out[13]: 
a    10
b    11
c    12
d    13
e    14
Name: three, dtype: int32

data.tail(1)   #返回DataFrame中的最后一行
data.head(1)   #返回DataFrame中的第一行
```
[python中pandas库中DataFrame对行和列的操作使用方法](http://blog.csdn.net/xiaodongxiexie/article/details/53108959 "python中pandas库中DataFrame对行和列的操作使用方法")

#### 10.03 同一坐标中多个hist 
```
# bins 设置分组的个数
# normed 是否对y轴数据进行标准化（如果true，则是在本区间的点在所有的点中所占的概率，False，则是显示点的数量）

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True) # 2行2列的直方图 共享坐标
for i in range(2):
     for j in range(2):
          ax[i, j].hist(np.random.randn(1000), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()
```
