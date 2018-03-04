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

### 04 矩阵和状态变化
#### 04.01 卡尔曼滤波器
![卡尔曼滤波器](https://i.imgur.com/8g0dPXh.png)

符号 | 含义 | 
- | :-: | -: 
x | 移动前 x 的位置|
x' | 移动后 x 的位置 |
v | x 方向的速度 |
delta t | 移动的持续时间“delta t” |


变量：状态，能反应物理角色的状态。
分为：
可观察物：瞬间位置。
隐藏对象：速度。
#### 04.01 简化卡尔曼滤波器方程
###### Kalman Filter Equations Fx Versus Bu
Consider this specific Kalman filter equation: x' = Fx + Bx

This equation is the move function that updates your beliefs in between sensor measurements. Fx models motion based on velocity, acceleration, angular velocity, etc of the object you are tracking.

B is called the control matrix and u is the control vector. Bu measures extra forces on the object you are tracking. An example would be if a robot was receiving direct commands to move in a specific direction, and you knew what those commands were and when they occurred. Like if you told your robot to move backwards 10 feet, you could model this with the Bu term.

When you take the self-driving car engineer nanodegree, you'll use Kalman filters to track objects that are moving around your vehicle like other cars, pedestrians, bicyclists, etc. In those cases, you would ignore BuBu because you do not have control over the movement of other objects. The Kalman filter equation becomes x' = Fx

[卡尔曼方程参考](https://classroom.udacity.com/courses/ud953/lessons/4632564251/concepts/555e0c78-8acd-4eda-acf7-4d794be34978 "卡尔曼方程参考")