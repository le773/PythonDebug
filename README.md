# 0.0 UdaMachineLearn

udacity machine learn term one record


### 1.0 如何将Anaconda更新到想要的python版本
重要事项说明：用以上方法前，请测试一下你的机器是否可以用anaconda更新（原生的Python，用Pip装东西也是同理，是否可以连接那些pip源。）
当确认可以后，做如下操作：
1. 点击Anaconda Prompt
2. 输入命令：
    conda update conda ——等待更新完，然后输入以下命令，这个较快
    conda create -n py34 python=3.4 anaconda ——等待更新完，然后输入以下命令。（因为要把python3.4的相关包下载下来，所以比较慢，请耐心等待）
    activate py34 # 切换环境
3. that's all

##### 注销该环境
    deactivate