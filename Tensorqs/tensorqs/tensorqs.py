'''
@Project ：PytorchTutorials 
@File    ：tensorqs.py.py
@Author  ：hailin
@Date    ：2022/10/24 21:47 
@Info    : https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
'''

import torch
import numpy as np

# 张量是一种特殊的数据结构，与数组和矩阵非常相似。在 PyTorch 中，我们使用张量对模型的输入和输出以及模型的参数进行编码。
# 张量类似于NumPy 的ndarray，除了张量可以在 GPU 或其他硬件加速器上运行。张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据。张量也针对自动微分进行了优化

# 张量可以直接从数据中创建。数据类型是自动推断的。
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("x_data:\n", x_data)

# 张量可以从 NumPy 数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("x_np:\n ", x_np)

# 从另一个张量保留参数张量的属性 形状 数据类型
x_ones = torch.ones_like(x_data)  # 从x_data而来
print(f"Ones Tensors:\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖量 datatype 跟x_data一样形状的浮点型随机的tensor
print(f"Random Tensors:\n{x_rand}\n")

# 使用随机或恒定值创建 shape是张量维度的元组，它绝对量输出张量的维度
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"rand tensor:\n{rand_tensor}\n")
print(f"ones tensor:\n{ones_tensor}\n")
print(f"zeros tensor:\n{zeros_tensor}\n")

# 张量的属性 描述他们的形状 数据类型 和存储他们的设备
tensor = torch.rand(3, 4)
print(f"shape:\n{tensor.shape}\n")
print(f"dtype:\n{tensor.dtype}\n")
print(f"device:\n{tensor.device}\n")

# 默认情况下，张量是在 CPU 上创建的。需要使用 .to方法明确地将张量移动到 GPU（在检查 GPU 可用性之后）。跨设备复制大张量在时间和内存方面可能会消耗很大！
print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     tensor=torch.to("cuda")

# 类似numpy的索引和切片
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 张量 连接 沿给定维度连接一系列张量
t1 = torch.cat([tensor, tensor], dim=1)
print("t1:", t1)

# 张量运算
print("tensor", tensor)
# 矩阵乘法
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
# y1 y2 y3 结果均相同
print("y1:", y1)
print("y2:", y2)
print("y3:", y3)
# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 单元素张量如果您有一个单元素张量，例如通过将张量的所有值聚合为一个值，您可以使用以下方法将其转换为 Python 数值item()：
print("tensor", tensor)
agg = tensor.sum()
print(agg, type(agg))
agg_item = agg.item()
print(agg_item, type(agg_item))

# 将结果存储到操作数中的操作称为就地操作 它们由_后缀表示。例如：x.copy_(y), x.t_(), 会变x。
# 可以节省一些内存，但在计算导数时可能会出现问题，因为会立即丢失历史记录
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

# 与Numpy连接
# 张量到 NumPy 数组
t = torch.ones(5)
print(f"t:{t},type:{type(t)}")
n = t.numpy()
print(f"n:{n},type:{type(n)}")

# 运算
t.add_(1)
print(f"t:{t}")
print(f"n:{n}")

# numpy 数组到张量
n=np.ones(5)
print(f"n:{n},type:{type(n)}")
t=torch.from_numpy(n)
print(f"t:{t},type:{type(t)}")

# 运算
np.add(n, 1, out=n) # 没有 结尾的_
print(f"t: {t}")
print(f"n: {n}")