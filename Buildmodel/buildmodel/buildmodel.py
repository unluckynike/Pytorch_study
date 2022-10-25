'''
@Project ：PytorchTutorials 
@File    ：buildmodel.py.py
@Author  ：hailin
@Date    ：2022/10/25 15:46 
@Info    : https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# torch.nn命名空间提供了构建自己的神经网络所需的所有构建块。
# PyTorch中的每个模块都是 nn.Module 的子类。神经网络是一个模块本身，它由其他模块（层）组成。这种嵌套结构允许轻松构建和管理复杂的架构。

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 通过子类定义我们的神经网络nn.Module，并在 中初始化神经网络层__init__。每个nn.Module子类都在方法中实现对输入数据的操作forward。

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 创建model 实例
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class:{y_pred}")

# 让我们分解 FashionMNIST 模型中的层。为了说明这一点，我们将抽取 3 张大小为 28x28 的图像的小批量样本，看看当我们通过网络传递它时会发生什么。
input_image = torch.rand(3, 28, 28)
print("input_image:\n", input_image.size())

# 初始化nn.Flatten 层以将每个 2D 28x28 图像转换为 784 个像素值的连续数组（保持小批量维度（dim=0））
flatten = nn.Flatten()
flat_image = flatten(input_image)
print("flat_image:\n", flat_image.size())

# nn.Linear 是一个模块，它使用其存储的权重和偏差对输入应用线性变换。
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print("hidden1:\n", hidden1.size())

# nn.ReLu 非线性激活是在模型的输入和输出之间创建复杂映射的原因。它们在线性变换后应用以引入非线性，帮助神经网络学习各种现象。
# 在这个模型中，我们在线性层之间使用nn.ReLU，但是还有其他激活可以在模型中引入非线性
print(f"Before ReLu:{hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLu:{hidden1}")

# nn.Sequential是一个有序的模块容器。数据按照定义的顺序通过所有模块。可以使用顺序容器来组合一个快速网络
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
input_image=torch.rand(3,28,28)
logits=seq_modules(input_image)

# nn.Softmax
# 神经网络的最后一个线性层返回logits - [-infty, infty] 中的原始值 - 被传递给 nn.Softmax模块。logits 被缩放为值 [0, 1]，表示模型对每个类别的预测概率。dim参数指示值必须总和为 1 的维度。
softmax=nn.Softmax(dim=1)
pred_probab=softmax(logits)

# 模型参数
# 神经网络内的许多层都是参数化的，即具有在训练期间优化的相关权重和偏差。
# 子类nn.Module化会自动跟踪模型对象中定义的所有字段，并使用模型parameters()或named_parameters()方法使所有参数都可以访问。
# 遍历每个参数，并打印其大小和其值的预览。
print(f"Model structre:{model}\n\n")
for name,param in model.named_parameters():
    print(f"Layer:{name} | Size:{param.size()} | Values:{param[:2]} \n")

