#### 1.0 如何将Anaconda更新到想要的python版本
重要事项说明：用以上方法前，请测试一下你的机器是否可以用anaconda更新（原生的Python，用Pip装东西也是同理，是否可以连接那些pip源。）
当确认可以后，做如下操作：
1. 点击Anaconda Prompt
2. 输入命令：
    conda update conda ——等待更新完，然后输入以下命令，这个较快
    conda create -n py34 python=3.4 anaconda ——等待更新完，然后输入以下命令。（因为要把python3.4的相关包下载下来，所以比较慢，请耐心等待）
    activate py34 # 切换环境
3. that's all

#### 1.1 注销该环境
```
deactivate
```
#### 1.2 修改主题颜色
```
pip install jupyterthemes
# 切换chesterish主题
jt -t chesterish
# 切换chesterish主题 + font
jt -f inconsolata -t chesterish
```
[jupyter-themes](https://github.com/dunovank/jupyter-themes "jupyter-themes")

#### 备注 希腊字母表及读音
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