
'''
@Project ：PytorchTutorials 
@File    ：data.py.py
@Author  ：hailin
@Date    ：2022/10/25 09:21 
@Info    : https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

# PyTorch 提供了两个数据原语：torch.utils.data.DataLoader允许torch.utils.data.Dataset 使用预加载的数据集以及您自己的数据。
# Dataset存储样本及其对应的标签，并DataLoader在 周围包裹一个可迭代对象Dataset，以便轻松访问样本。


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader


# Fashion-MNIST 是 Zalando 文章图像的数据集，由 60,000 个训练示例和 10,000 个测试示例组成。每个示例都包含 28×28 灰度图像和来自 10 个类别之一的相关标签。
training_data=datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())


# 可视化数据
labels_map={
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure=plt.figure(figsize=(8,8))
cols,rows=3,3
for i in  range(1,cols*rows+1):
    sample_idx=torch.randint(len(training_data),size=(1,)).item()
    img,label=training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()


# 创建自定义数据集
# 自定义 Dataset 类必须实现三个函数：__init__、__len__和__getitem__。
# FashionMNIST 图像存储在一个目录img_dir中，它们的标签分别存储在一个 CSV 文件annotations_file中。


class CustomImageDataset(Dataset):
    # __init__ 函数在实例化 Dataset 对象时运行一次。我们初始化包含图像、注释文件和两种转换的目录
    def __init__(self,annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform

    # __len__ 函数返回我们数据集中的样本数。
    def __len__(self):
        return len(self.img_labels)

    # __getitem__ 函数从给定索引处的数据集中加载并返回一个样本idx。基于索引，它识别图像在磁盘上的位置，
    # 将其转换为张量read_image，从 csv 数据中检索相应的标签self.img_labels，调用它们的转换函数（如果适用），并返回张量图像和相应的标签一个元组。
    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,o])
        image=read_image(img_path)
        label=self.img_labels.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image,label

train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

# 遍历 DataLoader
# 我们已将该数据集加载到DataLoader并且可以根据需要遍历数据集。
# 下面的每次迭代都会返回一批train_features和train_labels（分别包含batch_size=64特征和标签）。
# 因为我们指定shuffle=True了 ，所以在我们遍历所有批次之后，数据被打乱

train_features,train_labels=next(iter(train_dataloader))
img=train_features[0].squeeze()
label=train_labels[0]
print(f"Feature batch shape:{train_features.size()}")
print(f"Labels batch shape:{train_labels.size()}")
print(f"Label:{label}")
plt.imshow(img,cmap="gray")
plt.show()












