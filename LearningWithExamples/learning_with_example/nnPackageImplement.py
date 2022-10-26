'''
@Project ：PytorchTutorials 
@File    ：nnPackageImplement.py
@Author  ：hailin
@Date    ：2022/10/26 09:02 
@Info    : 
'''
import torch
import math

# use the nn package to implement our polynomial model network:

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 对于这个例子，输出y是(x, x^2, x^3)的线性函数， 我们可以将其视为线性层神经网络。
# 让我们准备 # 张量 (x, x^2, x^3)。
p = torch.tensor([1, 2, 3])
# x.unsqueeze(-1) 的形状为 (2000, 1)，p 的形状为 # (3,)，对于这种情况，将应用广播语义来获得张量
xx = x.unsqueeze(-1).pow(p)

# 使用 nn 包将我们的模型定义为一系列层。 nn.Sequential
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(f"t:{t} loss:{loss}")

    # 在运行反向传递之前将梯度归零。
    model.zero_grad()
    # backward pass：计算关于所有可学习的损失的梯度
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]
print(
    f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
