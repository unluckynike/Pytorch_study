'''
@Project ：PytorchTutorials 
@File    ：InitInstance.py
@Author  ：hailin
@Date    ：2022/10/26 16:06 
@Info    : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
            Initiate an instance
            Run the model
'''
import math

import torch.cuda
import copy
import time
from DataSet import vocab, bptt, train_data, get_batch, val_data, test_data
import TransformerModel
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# 运行模型
criterion = nn.CrossEntropyLoss()
learning_rate = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 每 step_size epochs 衰减每个参数组的学习率。请注意，这种衰减可能与此调度程序外部对学习率的其他更改同时发生。当 last_epoch=-1 时，设置初始 lr 为 lr。
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0
    log_interval = 200
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        # 训练期间，使用nn.utils.clip_grad_norm_ 来防止梯度爆炸。
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            learning_rate = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            # epoch=3
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {learning_rate:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')


def evalute(model: nn.Module, eval_data: Tensor) -> Tensor:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


# 循环遍历时代。如果验证损失是我们迄今为止看到的最好的，则保存模型。在每个 epoch 后调整学习率。
best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1,epochs+1):
    epoch_start_time=time.time()
    train(model)
    val_loss=evalute(model,val_data)
    val_ppl=math.exp(val_loss)
    elapsed=time.time()-epoch_start_time
    print('-'*89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss<best_val_loss:
        best_val_loss=val_loss
        best_model=copy.deepcopy(model)
    scheduler.step()

# Evaluate the best model on the test dataset
test_loss=evalute(best_model,test_data)
test_ppl=math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)