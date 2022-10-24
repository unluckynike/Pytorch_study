'''
@Project ：PytorchTutorials 
@File    ：quickstart.py.py
@Author  ：hailin
@Date    ：2022/10/24 20:26 
@Info    : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

# PyTorch 有两个处理数据的原语： torch.utils.data.DataLoader和torch.utils.data.Dataset.
# Dataset存储样本及其对应的标签，并且DataLoader在Dataset.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 流程 1处理数据  2创建模型or加载模型（提前保存） 3优化模型参数 4保存模型

# 处理数据
# 下载训练数据
training_data = datasets.FashionMNIST(root="data",  # root是存储训练/测试数据的路径，
                                      train=True,  # train指定训练或测试数据集，
                                      download=True,  # download=True如果数据不可用，则从 Internet 下载数据root。
                                      transform=ToTensor(),  # transform并target_transform指定特征和标签转换
                                      )

# 下载测试数据
test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  )

# Visualizing the Dataset 可视化数据集
# labels_map={
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure=plt.figure(figsize=(8,8))
# cols=3
# rows=3
# for i in range(1,cols*rows+1):
#     sample_idx=torch.randint(len(training_data),size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# 将Dataset作为参数传递给DataLoader。这对我们的数据集进行了迭代，并支持自动批处理、采样、混洗和多进程数据加载。
# 这里我们定义了一个64的batch size，即dataloader iterable中的每个元素都会返回一个batch 64个特征和标签。

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"shape of X[N,C,H,W ]:  {X.shape}")
    print(f"shape of y : {y.shape} {y.dtype}")
    break

# 创建模型
# 查看设备 使用cpu 或者 gpu 训练
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# 定义模型
# 定义一个继承自nn.Module的类 init函数定义网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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


model = NeuralNetwork().to(device)
print(model)

# 损失函数 优化模型参数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 在单个训练循环中，模型对训练数据集进行预测（分批输入），并反向传播预测误差以调整模型的参数。
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测错误
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss:{loss:>7f} [{current:>5f}/{size:>5d}]")


# 根据测试数据集检查模型的性能 确保她在学习
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch{t + 1}\n-------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done! ")

# 保存模型
torch.save(model.state_dict(),"model.path")
print("Save PyTorch Model State to Model.path")
#  加载模型
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))

# 模型预测
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 模型进行预测
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
