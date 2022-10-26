'''
@Project ：PytorchTutorials 
@File    ：DataSet.py
@Author  ：hailin
@Date    ：2022/10/26 15:22 
@Info    : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
           Load and batch data
           Functions to generate input and target sequence
'''

# 加载和批处理数据

# 用于torchtext生成 Wikitext-2 数据集。要访问 torchtext 数据集，按照https://github.com/pytorch/data上的说明安装 torchdata
# pip install torchdata
# vocab 对象是基于训练数据集构建的，用于将标记数字化为张量。Wikitext-2 将稀有标记表示为。
# 给定一个顺序数据的一维向量，batchify()将数据排列成batch_size列。如果数据没有均匀地分成 batch_size列，那么数据将被修剪以适应。
# 批处理可以实现更多的并行化处理。但是，批处理意味着模型独立处理每一列
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from torch import Tensor
from typing import Tuple

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(row_text_iter: dataset.IterableDataset) -> Tensor:
    """
    Converts raw text into a flat Tensor.
    将原始文本转换为平面张量
    :param row_text_iter:
    :return:
    """
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in row_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data: Tensor, bsz: int) -> Tensor:
    """
    将数据分成bsz个单独的序列，删除不完全匹配的额外元素。
    :param data:
    :param bsz: batch_size
    :return:  Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# 生成输入和目标序列的函数
# get_batch()为转换器模型生成一对输入-目标序列。它将源数据细分为长度的块bptt。
# 对于语言建模任务，模型需要以下单词 as Target。例如，bptt值为 2 时，我们将得到以下两个i= 0 的变量：
bptt = 35


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
      source: Tensor, shape [full_seq_len, batch_size]
        i: int
    :param source:
    :param i:
    :return: tuple (data, target), where data has shape [seq_len, batch_size] and
             target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target
