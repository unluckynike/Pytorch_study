
'''
@Project ：PytorchTutorials 
@File    ：save_model.py.py
@Author  ：hailin
@Date    ：2022/10/25 21:12 
@Info    : https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
'''
import torch
import torchvision.models as models # 保存模型

# PyTorch 模型将学习到的参数存储在内部状态字典中，称为state_dict. 这些可以通过以下torch.save 方法持久化：
# 保存模型
model=models.vgg16(pretrained=True)
torch.save(model.state_dict(),'model_weights.path')

# 要加载模型权重，首先需要创建一个相同模型的实例，然后使用load_state_dict()方法加载参数。
# 加载模型
model=models.vgg16()
# load
model.load_state_dict(torch.load('model_weights.path'))
model.eval()
