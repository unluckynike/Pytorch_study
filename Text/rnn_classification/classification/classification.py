
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

print(category_lines['Italian'][:5])


