
'''
@Project ：Pytorch_study 
@File    ：classification.py.py
@Author  ：hailin
@Date    ：2022/10/27 20:32 
@Info    : https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
          rnn 姓名分类 训练来自 18 种语言的几千个姓氏，并根据拼写预测一个名字来自哪种语言：
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path):
    return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters=string.ascii_letters+".,;'"
n_letters=len(all_letters)

# 将unicode字符转换为ASCII
def unicodeToAscii(s):
    return  ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn' and c in all_letters)

print(unicodeToAscii('Ślusàrski'))

# 建立分类字典
category_lines={}
all_categories=[]

# 读取并拆分文件
def readLines(filename):
    lines=open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category=os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines=readLines(filename)
    category_lines[category]=lines

n_categories=len(all_categories)

# 现在有了category_lines，一个将每个类别（语言）映射到行（名称）列表的字典。
# 还记录了 all_categories（只是语言列表）并n_categories供以后参考。

print(category_lines['Italian'][:10])

# 现在我们已经组织好了所有的名称，我们需要将它们变成张量来使用它们。
import torch

# 把名字变成张量
# 从all_letters找到索引 例如 a=0
def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor=torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)]=1
    return tensor

def lineToTensor(line):
    tensor=torch.zeros(len(line),1,n_letters)
    for li,letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)]=1
    return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size())

# 创建模型
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()

        self.hidden_size=hidden_size

        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(input_size+hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        combined=torch.cat((input,hidden),1) #  按维数1拼接（横着拼） 0维则是竖着拼接
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        output=self.softmax(output)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

n_hidden=128
rnn=RNN(n_letters,n_hidden,n_categories)

# 传递一个输入（在我们的例子中，当前字母的张量）和一个先前的隐藏状态（我们首先将其初始化为零）。
# 我们将取回输出（每种语言的概率）和下一个隐藏状态（我们为下一步保留）。

input=letterToTensor('A')
hidden=torch.zeros(1,n_hidden)
output,next_hidden=rnn(input,hidden)

# 为了提高效率，我们不为每一步都创建一个新的张量，将使用切片lineToTensor来代替 letterToTensor和使用切片。
# 这可以通过预先计算成批的张量来进一步优化。

input=lineToTensor('Albert')
hidden=torch.zeros(1,n_hidden)

output,next_hidden=rnn(input[0],hidden)
# 输出是一个张量，其中每个都是该类别的可能性（越高越有可能）
print(output)

