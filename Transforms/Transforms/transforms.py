'''
@Project ：PytorchTutorials 
@File    ：transforms.py.py
@Author  ：hailin
@Date    ：2022/10/25 14:56 
@Info    : https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
'''

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

#所有 TorchVision 数据集都有两个参数 -transform修改特征和 target_transform修改标签 - 接受包含转换逻辑的可调用对象。
# torchvision.transforms模块提供了几个开箱即用的常用转换。
ds = datasets.FashionMNIST(root="data",
                           train=True,
                           download=True,
                           transform=ToTensor(),
                           # Lambda 转换应用任何用户定义的 lambda 函数。在这里定义了一个函数来将整数转换为 one-hot 编码张量。
                           # 它首先创建一个大小为 10 的零张量（我们数据集中的标签数量）并调用 scatter_，它在标签上分配 a value=1给定的索引y
                           target_transform=Lambda(
                               lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                           )
