'''
@Project ：PytorchTutorials 
@File    ：PositionalEncoding.py
@Author  ：hailin
@Date    ：2022/10/26 14:46 
@Info    : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
import math

import torch
from torch import nn, Tensor

# 定义模型

# 模块注入一些关于序列中标记的相对或绝对位置的信息。位置编码与嵌入具有相同的维度，因此可以将两者相加。在这里使用不同频率的sine和cosine函数。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropoout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropoout)

        postion = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:0, 0, 0::2] = torch.sin(postion * div_term)
        pe[:0, 0, 1::2] = torch.cos(postion * div_term)
        self.register_buffer("pe", pe)

        def forward(self, x: Tensor) -> Tensor:
            """

            :param self:
            :param x: Tensor, shape [seq_len, batch_size, embedding_dim]
            :return:
            """
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
