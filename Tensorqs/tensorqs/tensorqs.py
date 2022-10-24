
'''
@Project ：PytorchTutorials 
@File    ：tensorqs.py.py
@Author  ：hailin
@Date    ：2022/10/24 21:47 
@Info    : https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
'''

import  torch
import numpy as np

# 张量是一种特殊的数据结构，与数组和矩阵非常相似。在 PyTorch 中，我们使用张量对模型的输入和输出以及模型的参数进行编码。
# 张量类似于NumPy 的ndarray，除了张量可以在 GPU 或其他硬件加速器上运行。张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据。张量也针对自动微分进行了优化


