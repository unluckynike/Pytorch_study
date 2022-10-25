'''
@Project ：PytorchTutorials 
@File    ：autogradqs.py.py
@Author  ：hailin
@Date    ：2022/10/25 16:26 
@Info    : https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
'''
import torch

# 在训练神经网络时，最常用的算法是 反向传播。在该算法中，参数（模型权重）根据损失函数相对于给定参数的梯度进行调整。
# 为了计算这些梯度，PyTorch 有一个内置的微分引擎，称为torch.autograd. 它支持任何计算图的梯度自动计算。
# 考虑最简单的一层神经网络，具有输入x、参数w和b以及一些损失函数。它可以通过以下方式在 PyTorch 中定义：

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 计算梯度
# 为了优化神经网络中参数的权重，我们需要计算我们的损失函数对参数的导数，即我们需要 loss偏w的导数，loss偏b的导数x在和的一些固定值下的y。
# 为了计算这些导数，我们调用 loss.backward()，然后从w.grad和 检索值b.grad
loss.backward()
print("w.grad:\n", w.grad)
print("b.grad:\n", b.grad)

# 禁用梯度
z = torch.matmul(x, w) + b
print(z.requires_grad)  # True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)  # False

# 禁用梯度的另一种方法 detach()
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)  # False

print(torch.eye(4,5,requires_grad=True))
inp=torch.eye(4,5,requires_grad=True)
out=(inp+1).pow(2).t()# 所有元素+1 再平方 再转置
print("out:\n",out)
out.backward(torch.ones_like(out),retain_graph=True)
print(f"Fists call \n {inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nthird call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nforth call\n{inp.grad}")

inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")




