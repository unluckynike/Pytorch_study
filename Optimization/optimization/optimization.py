'''
@Project ：PytorchTutorials 
@File    ：optimization.py.py
@Author  ：hailin
@Date    ：2022/10/25 20:21 
@Info    : https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1 设置数据集 训练集 测试集
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

batch_size = 64
# 2 设置dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# 2 定义模型 记得要实例话
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 4 实例化模型  设置超参数 初始化损失函数和优化器
model = NeuralNetwork()
# 超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值会影响模型训练和收敛速度
# Number of Epochs - 迭代数据集的次数
# Batch Size - 参数更新前通过网络传播的数据样本数
# 学习率- 在每个批次/时期更新模型参数的程度。较小的值会产生较慢的学习速度，而较大的值可能会导致训练期间出现不可预测的行为。
learning_rate = 1e-3
batch_size = 64
epochs = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 调用optimizer.zero_grad()以重置模型参数的梯度。默认情况下渐变加起来；为了防止重复计算，我们在每次迭代时明确地将它们归零。
# 通过调用来反向传播预测损失loss.backward()。PyTorch 存储每个参数的损失梯度。
# 一旦我们有了我们的梯度，我们调用optimizer.step()通过在反向传递中收集的梯度来调整参数。

# 定义train_loop循环优化代码，并test_loop根据我们的测试数据评估模型的性能
# 5 定义循环优化训练集
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 6 定义测试评估
def test_loop(dataloader, model, loss_fn):  # 测试集就不优化了
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 优化是在每个训练步骤中调整模型参数以减少模型误差的过程。优化算法定义了如何执行这个过程（在这个例子中，我们使用随机梯度下降）。
# 所有优化逻辑都封装在optimizer对象中。在这里，我们使用 SGD 优化器；此外，PyTorch 中有许多不同的优化器 可用，例如 ADAM 和 RMSProp，它们可以更好地用于不同类型的模型和数据。

for t in range(epochs):
    print(f"Epochs:{t + 1}\n-------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)  # 测试集就不需要优化了
print("Done! ")


