#### 05.12 正弦 余弦 曲线
```
def sin_cos():
    num_points = 50
    x = np.zeros(num_points)
    sin_x = np.zeros(num_points)
    cos_x = np.zeros(num_points)

    for i in range(num_points):
        x[i] = 2. * math.pi * i / (num_points - 1.)
        sin_x[i] = math.sin(x[i])
        cos_x[i] = math.cos(x[i])
    return x, sin_x, cos_x


x, sin_x, cos_x = sin_cos()

plt.plot(x, sin_x)
plt.plot(x, cos_x)

plt.show()
```
#### 05.13 表面重力 
```
G=mg，g为比例系数，重力大小约为9.8N/kg
```

#### 05.14 欧拉折线法
![欧拉折线法](https://pic3.zhimg.com/v2-8e271c6789e0fb747ff86c391d5c6c97_r.jpg)

#### 05.16 矢量概述
矢量相加
矢量相减(减的那个是反方向)

#### 05.18 牛顿引力
```
F = (M1 * M2) * G / (d * d) = M1 * a
```

#### 05.20 蛋壳定理

