'''
@Project ：PytorchTutorials 
@File    ：sentiment.py.py
@Author  ：hailin
@Date    ：2022/10/27 08:29 
@Info    : https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
           文本分类
'''

# 访问原始数据集迭代器
# torchtext 库提供了一些原始数据集迭代器，它们产生原始文本字符串。例如，AG_NEWS数据集迭代器将原始数据生成为标签和文本的元组。
import time

import torch
from torchtext.datasets import AG_NEWS

print("start")
train_iter = iter(AG_NEWS(split='train'))

# 准备数据集
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab(['here', 'is', 'an', 'example'])
vocab.set_default_index(vocab["<unk>"])

text_pipline = lambda x: vocab(tokenizer(x))
label_pipline = lambda x: int(x) - 1

# 生成数据批处理和迭代器
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  collate_fn (callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
#                                  Used when using batched loading from a map-style dataset.
# collate_fn函数对从DataLoader. 的输入collate_fn是一批批大小为 的数据DataLoader，并collate_fn根据之前声明的数据处理管道对其进行处理
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipline(_label))
        process_text = torch.tensor(text_pipline(_text), dtype=torch.int64)
        text_list.append(process_text)
        offsets.append(process_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# 定义模型
from torch import nn


# 该模型由nn.EmbeddingBag层和用于分类目的的线性层组成。

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        :param vocab_size: 整个语料包含的不同词汇总数
        :param embed_dim: 指定词嵌入的维度
        :param num_class: 文本分类的类别总数
        """
        super(TextClassificationModel, self).__init__()
        # nn.EmbeddingBag使用默认模式“mean”计算嵌入“bag”的平均值。尽管此处的文本条目具有不同的长度，但 nn.EmbeddingBag 模块在此处不需要填充，因为文本长度保存在偏移量中。
        # 此外，由于nn.EmbeddingBag动态累积嵌入的平均值，nn.EmbeddingBag可以提高性能和内存效率以处理一系列张量。
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # 各层的权重参数都是初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def foward(self, text, offsets):
        embeded = self.embedding(text, offsets)
        return self.fc(embeded)


# 初始化模型
train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# 定义函数训练模型
import time
from torch.optim import optimizer


def train(model):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 100
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch{:3d} | {:5d}/{:5d} batches | accuracy{:8.3f}'.format(epoch, idx, len(dataloader),
                                                                                total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# 拆分数据集 运行模型
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

EPOCHS = 10
LR = 5
BATCH_SIZE = 64

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# StepLR 在这里用于通过 epochs 调整学习率。
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
# 测试集
test_datasets = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
# 由于原始AG_NEWS数据集没有有效数据集，将训练数据集分成训练/有效集，分割比率为 0.95（训练）和 0.05（有效）。
# 使用 PyTorch 核心库中的 torch.utils.data.dataset.random_split 函数。
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu=accu_val
    print('-'*59)
    print('|and of epoch {:3d}|time: {:5.2f}s |valid accuracy {:8.3d}'.format(epoch,time.time()-epoch_start_time,accu_val))
    print('-'*59)


#   检查测试数据集的结果
print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))


# 测试随机新闻
ag_news_label={
    1:"World",
    2:"Sport",
    3:"Bussiness",
    4:"Sci/Tec"
}

def predict(text,text_pipeline):
    with torch.no_grad():
        text=torch.tensor(text_pipeline(text))
        output=model(text,torch.tensor([0]))
        return output.argmax(1).item()+1

# ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
#     enduring the season’s worst weather conditions on Sunday at The \
#     Open on his way to a closing 75 at Royal Portrush, which \
#     considering the wind and the rain was a respectable showing. \
#     Thursday’s first round at the WGC-FedEx St. Jude Invitational \
#     was another story. With temperatures in the mid-80s and hardly any \
#     wind, the Spaniard was 13 strokes better in a flawless round. \
#     Thanks to his best putting performance on the PGA Tour, Rahm \
#     finished with an 8-under 62 for a three-stroke lead, which \
#     was even more impressive considering he’d never played the \
#     front nine at TPC Southwind."

ex_text_str="Dialogue systems, also called chatbots, are now used in a wide range of applications." \
            " However, they still have some major weaknesses. One key weakness is that they are typi- cally trained " \
            "from manually-labeled data and/or written with handcrafted rules, and their knowledge bases (KBs) are" \
            " also compiled by human experts. Due to the huge amount of man- ual effort involved, they are difficult to" \
            " scale and also tend to produce many errors ought to their limited ability to un- derstand natural language and" \
            " the limited knowledge in their KBs. Thus, the level of user satisfactory is often low. In this paper, we propose " \
            "to dramatically improve the situation by endowing the chatbots the ability to continually learn (1) new world knowledge, " \
            "(2) new language expressions to ground them to actions, and (3) new conversational skills, during con- versation by " \
            "themselves so that as they chat more and more with users, they become more and more knowledgeable and are better and " \
            "better able to understand diverse natural lan- guage expressions and to improve their conversational skills."


model=model.to('cpu')
print("This is a %s new "%ag_news_label[predict(ex_text_str,text_pipline)])






