'''
@Project ：PytorchTutorials 
@File    ：nn.py.py
@Author  ：hailin
@Date    ：2022/10/26 09:24 
@Info    : https://pytorch.org/tutorials/beginner/nn_tutorial.html
'''

from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import torch
import numpy as np
import math
from IPython.core.debugger import set_trace
import torch.nn.functional as F

# 使用pathlib 处理路径（Python 3 标准库的一部分），并将使用 requests下载数据集。
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# 使用 pickle 存储，pickle 是一种用于序列化数据的特定于 python 的格式。
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# 每个图像为 28 x 28，并存储为长度为 784 (=28x28) 的扁平行。我们来看一个；我们需要先将其重塑为 2d。
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(f"x_train.shape:{x_train.shape}")

# PyTorch 使用torch.tensor，而不是 numpy 数组，所以我们需要转换我们的数据。
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
print("x_train:\n", x_train)
print("y_train:\n", y_train)
print("x_train.shape:\n", x_train.shape)
print("y_train.min()\n", y_train.min())
print("y_train.max():\n", y_train.max())

# 从头开始 没有torch.nn的神经网络
print("-----------从头开始的神经网络（没有 torch.nn）----------------")
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


# @代表矩阵乘法运算
def model(xb):
    return log_softmax(xb @ weights + bias)


batch_size = 64

xb = x_train[0:batch_size]  # a mini-batch from x
preds = model(xb)  # predictions
print(f"preds[0]:\n {preds[0]} preds.shape:\n{preds.shape}")  # preds张量不仅包含张量值，还包含梯度函数


# 实现负对数似然作为损失函数
def null(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = null

# 我们的随机模型检查我们的损失，这样我们就可以看看我们在稍后通过反向传播后是否有所改善。
yb = y_train[0:batch_size]
print(f"loss_func:\n{preds, yb}")


# 实现一个函数来计算我们模型的准确性。对于每个预测，如果具有最大值的索引与目标值匹配，则预测是正确的。
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


print(f"准确率 accuracy(preds,yb): {accuracy(preds, yb)}")
"""
运行一个训练循环。对于每次迭代将：
  - 选择一小批数据（大小bs）
  - 使用模型进行预测
  - 计算损失
  - loss.backward()更新模型的梯度，在这种情况下，weights 并且bias.
"""
learning_rate = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n - 1) // batch_size + 1):
        # set_trace()
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * learning_rate
            bias -= bias.grad * learning_rate
            weights.grad.zero_()
            bias.grad.zero_()

print(f"loss_func(model(xb), yb):\n{loss_func(model(xb), yb)}\n  accuracy(model(xb), yb):\n{accuracy(model(xb), yb)}")

print("------------使用 torch.nn.functional--------------")
loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(f"loss_func(model(xb), yb):\n{loss_func(model(xb), yb)}\n  accuracy(model(xb), yb):\n{accuracy(model(xb), yb)}")
