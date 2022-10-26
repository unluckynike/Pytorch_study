'''
@Project ：PytorchTutorials 
@File    ：TransformerModel.py.py
@Author  ：hailin
@Date    ：2022/10/26 11:06 
@Info    : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerDecoderLayer, TransformerEncoderLayer
from torch.utils.data import dataset
import PositionalEncoding

# 定义模型

# 首先将一系列标记传递给嵌入层，然后是位置编码层以说明单词的顺序
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super(self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder=TransformerEncoder(encoder_layers,nlayers)
        self.encoder=nn.Embedding(ntoken,d_model)
        self.decoder=nn.Linear(d_model,ntoken)

        self.init_weights()

    def init_weights(self) -> None: # ->None  增加代码可读性，告诉你返回的是一个None 的数据
        initrange=0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self,src:Tensor,src_mask:Tensor) -> Tensor:
        """

        :param src:
        :param src_mask:
        :return: output
        """
        src=self.encoder(src)*math.sqrt(self.d_model)
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src,src_mask)
        output=self.decoder(output)
        return output

    def generate_square_subsequent_mask(sz:int) -> Tensor:
        return torch.triu(torch.ones(sz,sz)*float('-inf'),diagonal=1)

