### 1.0 指数加权平均
#### 1.1 指数加权平均
![exponentially_weighted_average_2.png](https://i.imgur.com/FfV3x3f.png)

当`β`较大时，指数加权平均值适应的更慢一些。

#### 1.2 理解指数加权平均
![exponentially_weighted_average_1.png](https://i.imgur.com/WB4ErAA.png)

`Vt = β*(Vt-1) + (1-β)θt`，`θt`当天的实际气温，`(Vt-1)`前一天预测的气温。

![exponentially_weighted_average_3.png](https://i.imgur.com/1HHk3At.png)

当`β=0.9`时，相当于把过去`1/(1-β)`天的温度加权平均后，作为当日的气温。

#### 1.3 指数加权平均的偏差修正
![bias_weight_1.png](https://i.imgur.com/Yab1zTg.png)

当进行指数加权平均计算时，第一个值`V0`被初始化为0，这样将在前期运算产生一定的偏差。为了矫正偏差，需要用以上公式修正偏差。`t`越大时，修正效果越小。


