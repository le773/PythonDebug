### 1.0 双层神经网络

在网络里面添加一个隐藏层，可以让它构建更复杂的模型。而且，在隐藏层用非线性激活函数可以让它对非线性函数建模。

一个常用的非线性函数叫`ReLU（rectified linear unit）`。`ReLU`函数对所有负的输入，返回 0；所有 x >0 的输入，返回 x。

### 2.0 ReLUs
```
# tf.nn.relu() 放到隐藏层
# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# Hidden Layer with ReLU activation function
# 隐藏层用 ReLU 作为激活函数
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
output = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
```