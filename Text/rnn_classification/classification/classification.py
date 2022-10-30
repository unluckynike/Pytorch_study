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
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

def findFiles(path):
    return glob.glob(path)


print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)


# 将unicode字符转换为ASCII
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


print(unicodeToAscii('Ślusàrski'))

# 建立分类字典
category_lines = {}
all_categories = []


# 读取并拆分文件
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

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
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor('J'))
print(lineToTensor('Jones').size())

# 创建模型
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)  # 按维数1拼接（横着拼） 0维则是竖着拼接
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories).to(device)

# 传递一个输入（在我们的例子中，当前字母的张量）和一个先前的隐藏状态（我们首先将其初始化为零）。
# 我们将取回输出（每种语言的概率）和下一个隐藏状态（我们为下一步保留）。

input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden)

# 为了提高效率，我们不为每一步都创建一个新的张量，将使用切片lineToTensor来代替 letterToTensor和使用切片。
# 这可以通过预先计算成批的张量来进一步优化。

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
# 输出是一个张量，其中每个都是该类别的可能性（越高越有可能）
print(output)  # 对应的是18个txt类别


# 准备训练
# 在开始训练之前，我们应该创建一些辅助函数。首先是解释网络的输出，我们知道这是每个类别的可能性。可以使用Tensor.topk获取最大值的索引：
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


print(categoryFromOutput(output))

# 还需要一种快速获取训练示例（名称及其语言）的方法：
import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('categoty｜分类 = ', category, '        line｜名字 =', line)

# 训练网络
# 因为 RNN 的最后一层是nn.LogSoftmax，损失函数nn.NLLLoss是合适的，
criterion = nn.NLLLoss()

# 每个训练循环将：
# 创建输入和目标张量
# 创建一个归零的初始隐藏状态
# 阅读每个字母并为下一个字母保持隐藏状态
# 将最终输出与目标进行比较
# 反向传播
# 返回输出和损失

learing_rate = 0.005


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()  # return torch.zeros(1,self.hidden_size)
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learing_rate)

    return output, loss.item()


# 现在我们只需要用一些例子来运行它。由于该 train函数同时返回输出和损失，我们可以打印它的猜测并跟踪损失以进行绘图。
# 由于有 1000 个示例，我们只打印每个print_every示例，并取损失的平均值。

import time
import math

n_iters = 1000000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    # 打印 iter number loss name guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


# 输入预测
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
