'''
@Project ：PytorchTutorials 
@File    ：numpyImplement.py.py
@Author  ：hailin
@Date    ：2022/10/26 08:21 
@Info    : https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
'''

import numpy as np
import math

# 拟合 y=sin(x) 的问题作为我们的运行示例。该网络将有四个参数，并将使用梯度下降进行训练，通过最小化网络输出和真实输出之间的欧几里德距离来拟合随机数据。

# implement the network using numpy.

# 生存随机输入 输出数据
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 随机初始权重
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y=a+bx+cx^2+dx^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # loss
    loss = np.square(y_pred - y).sum() # 均方误差
    if t % 100 == 99:
        print(f"t:{t} loss:{loss}")

    # 反向传播计算 a、b、c、d 相对于损失的梯度
    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 更新权重
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Resylt:y={a}+{b}x+{c}x^2+{d}x^3")
