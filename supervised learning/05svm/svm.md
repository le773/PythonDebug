### 00 svm

### 02 超平面的方程
margin:越大越好
二次规范:奇偶
hyperplanes:超平面
![超平面公式](http://img.blog.csdn.net/20131107201104906)
- w:代表平面参数
- b:离开原点的因素
- y:分类标签

### 03 平面之间的距离
![超平面间的距离](https://i.imgur.com/Ez7dl4l.png)

最大化`2/(w.T)`转化为最小化`0.5 * ||w|| * ||w||`
平方项保证了表达式的单调性，它会放大结果但不会改变表达式的顺序。

### 06 最佳分离器
注意:
距离边界很远的点不能用来定义决策边界的轮廓.因为对结果没有影响.

找出所有的点对,弄清楚哪些点是重要的,能够影响决策边界的定义,然后思考从其输出标签的角度,它们如何彼此相关.

点积:
![点积](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D268/sign=d3491000c9fc1e17f9bf8b377291f67c/63d9f2d3572c11df390c1835652762d0f603c2c3.jpg)
向量a在向量b上投影的积,与余弦成正比;表示在相同方向上的相识度.

### 07 线性结合
### 08 内核
1. 
![径向基核](http://img.my.csdn.net/uploads/201304/03/1364958259_8460.jpg)
x1,x2越相近, K(x1,x2)越接近于1

2. ![多项式核](https://i.imgur.com/Daxm0UQ.png)
<x1,x2>:x1,x2的内积
k(x1,x2):x1 x2的相似性

3. 线性核
![线性核](http://img.my.csdn.net/uploads/201304/03/1364958354_7262.jpg)
实际上就是原始空间中的内积.

4. 双曲正切
![双曲正切](https://gss2.bdstatic.com/-fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D177/sign=adf59bb1f51fbe09185ec7135c610c30/96dda144ad345982648941550bf431adcaef84f2.jpg)