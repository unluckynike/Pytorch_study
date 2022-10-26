'''
@Project ：PytorchTutorials 
@File    ：TensorImplement.py
@Author  ：hailin
@Date    ：2022/10/26 08:43 
@Info    : 
'''

import torch
import math

# Numpy 是一个很棒的框架，但它不能利用 GPU 来加速其数值计算。对于现代深度神经网络，GPU 通常提供 50 倍或更高的加速，
# 使用 PyTorch 张量将三阶多项式拟合到正弦函数。 numpy 示例一样，需要手动实现通过网络的前向和后向传递

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(f"t:{t} loss:{loss}")

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Result:y={a.item()}+{b.item()}x+{c.item()}x^2+{d.item()}x^3")
