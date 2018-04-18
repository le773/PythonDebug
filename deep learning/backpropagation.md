## 利用反向传播训练多层神经网络的原理

此项目描述了利用反向传播算法的多层神经网络的教学过程,为了说明此过程，下图展示了有两个输入和一个输出的三层神经网络:

![bp01](https://i.imgur.com/h5uAE50.gif)

每个神经元由两个单元组成。第一个单元增加了产品的权重系数和输入信号。第二单元代表非线性函数，其被称为神经元激活函数。
`e`是权重与输入信号的累加和，然后作为非线性函数`f(e)`的形参。信号 `y`是此神经元的输出。

![bp02.gif](https://i.imgur.com/MJzWHXj.gif)

为了教授神经网络，我们需要训练数据集。训练数据集由输入信号（`x1`和`x2`）组成，并与相应的目标（期望的输出）`z`分配。网络培训是一个迭代的过程。在每次迭代中，使用来自训练数据集的新数据修改节点的权重系数。使用下面描述的算法计算的修改：每一次教学步骤都强制从训练集的输入信号开始。在这个阶段之后，我们可以确定每个网络层中每个神经元的输出信号值。下面的图片说明了信号是如何通过网络传播的，符号 `w(xm)n`表示输入层中网络输入`xm`和神经元`n`之间的连接权重。符号`yn`代表神经元`n`的输出信号。

![bp03.gif](https://i.imgur.com/QmLs6ar.gif)

![bp04.gif](https://i.imgur.com/dlPi05E.gif)

![bp05.gif](https://i.imgur.com/xQpf1Zg.gif)

通过隐藏层传播信号。符号`wmn`代表了神经元`m`与下一层神经元`n`之间的连接的权重。

![bp06.gif](https://i.imgur.com/GAuLlL4.gif)

![bp07.gif](https://i.imgur.com/voaSW75.gif)

通过输出层传播信号。

![bp08.gif](https://i.imgur.com/XcJ8giS.gif)

在下一个算法步骤中，网络`y`的输出信号与期望的（目标）输出值进行比较，其可以在训练集找到。这种差异被称为输出层神经元的错误信号`σ`。

![bp09.gif](https://i.imgur.com/7Ers9RD.gif)

要直接计算内部神经元的错误信号是不可能的，因为这些神经元的输出值是未知的。许多年来，人们对训练多人网络的有效方法并不知道。在80年代中期，只有反向传播算法才算出来。我们的想法是将错误信号`σ`（在单一教学步骤中计算）传播给所有的神经元，就是那些讨论神经元输入的输出信号。

![bp10.gif](https://i.imgur.com/3BpX1pQ.gif)

![img11.gif](https://i.imgur.com/ZTEZQWV.gif)

用于方向传播错误的权重系数`wmn`等于在计算输出值时使用的系数。只有数据流的方向发生了变化(信号从输出传播到另一个输入)。这种技术适用于所有网络层。如果传播的错误来自于它们被添加的几个神经元。下面图示:

![img12.gif](https://i.imgur.com/4JXEUCl.gif)

![img13.gif](https://i.imgur.com/SR2UuCs.gif)

![img14.gif](https://i.imgur.com/JxRDdi7.gif)

当每个神经元的错误信号已经计算好，每个神经元输入节点的权重系数就可以被修改。在下面的公式中，`df(e)/de`表示神经元激活函数的导数(其权重被修改)。

![img15.gif](https://i.imgur.com/OS9BK9C.gif)

![img16.gif](https://i.imgur.com/QnNQLRJ.gif)

![img17.gif](https://i.imgur.com/8ZwxttM.gif)

![img18.gif](https://i.imgur.com/jfy497D.gif)

![img19.gif](https://i.imgur.com/ioX5SqL.gif)

![bp20.gif](https://i.imgur.com/9CNRrXB.gif)

系数h影响网络学习速率。有几个选择学习速率的方法。
- 第一种，是使用大一点的值开始学习过程。当权重系数被建立时，参数会逐渐减小。
- 第二种，更复杂些，用一个小一点的值开始学习，在学习过程中，当学习效果逐步上升时，学习速率会增加，并在最后阶段会再次下降。用小的学习速率开始学习过程，可以确定权重系数的符号。

翻译自：[Principles of training multi-layer neural network using backpropagation](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html "Principles of training multi-layer neural network using backpropagation")