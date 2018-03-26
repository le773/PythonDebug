### 1.0 矩阵A奇异值分解
直观上，奇异值分解将矩阵分解为若干个秩一矩阵之和，用公式表示：
![矩阵A奇异值分解](https://www.zhihu.com/equation?tex=%281%29+%5Cquad%5Cquad+%5Cqquad+A+%3D+%5Csigma_1+u_1v_1%5E%7B%5Crm+T%7D%2B%5Csigma_2+u_2v_2%5E%7B%5Crm+T%7D%2B...%2B%5Csigma_r+u_rv_r%5E%7B%5Crm+T%7D)
其中等式右边每一项前的系数σ就是奇异值,μ和υ分别表示列向量，秩一矩阵的意思是矩阵秩为1，注意到每一项![uvt](https://www.zhihu.com/equation?tex=uv%5E%7B%5Crm+T%7D)都是秩为1的矩阵。假定奇异值满足![奇异值](https://www.zhihu.com/equation?tex=%5Csigma_1%5Cgeq%5Csigma_2%5Cgeq...%5Cgeq%5Csigma_r%3E0)。

### 1.0 奇异值的物理意义
奇异值往往对应着矩阵中隐含的重要信息，且重要性和奇异值大小正相关。每个矩阵A都可以表示为一系列秩为1的小矩阵之和，而奇异值则衡量了这些小矩阵对于A的权重。

### 奇异值的应用
在图像处理领域，奇异值不仅可以应用在**数据压缩**上，还可以对**图像去噪**。如果一副图像包含噪声，我们有理由相信那些较小的奇异值就是由于噪声引起的。当我们强行令这些较小的奇异值为0时，就可以去除图片中的噪声。

### 矩阵M的奇异值分解
![矩阵M奇异值分解](https://i.imgur.com/SJ3lm6r.png)

奇异值分解把线性变换清晰地分解为旋转、缩放、投影这三种基本线性变换。
![奇异值分解_旋转_缩放_投影](https://pic4.zhimg.com/v2-ea67bee7f332fa7bab9bb4ccf19f17e4_r.jpg)
[奇异值的物理意义是什么？](https://www.zhihu.com/question/22237507 "奇异值的物理意义是什么？")


###### 希腊字母表及读音
```
1 Α α alpha /a:lf/ 阿尔法 
2 Β β beta /bet/ 贝塔 
3 Γ γ gamma /ga:m/ 伽马 
4 Δ δ delta /delt/ 德尔塔 
5 Ε ε epsilon /ep`silon/ 伊普西龙
6 Ζ ζ zeta /zat/ 截塔 
7 Η η eta /eit/ 艾塔 
8 Θ θ thet /θit/ 西塔 
9 Ι ι iot /aiot/ 约塔 
10 Κ κ /kappa/ kap 卡帕 
11 ∧ λ /lambda/ lambd 兰布达 
12 Μ μ mu /mju/ 缪 
13 Ν ν nu /nju/ 纽 
14 Ξ ξ xi /ksi/ 克西 
15 Ο ο omicron /omik`ron/ 奥密克戎 
16 ∏ π pi /pai/ 派 
17 Ρ ρ rho /rou/ 柔
18 ∑ σ sigma /`sigma/ 西格马 
19 Τ τ tau /tau/ 套 
20 Υ υ upsilon /jup`silon/ 宇普西龙 
21 Φ φ phi /fai/ 佛爱 
22 Χ χ chi /phai/ 西 
23 Ψ ψ psi /psai/ 普西 
24 Ω ω omega /o`miga/ 欧米伽
```