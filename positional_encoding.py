import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        执行论文的PE方程:
            PE(pos,2i) = sin(pos/10000^(2i/d_model))
            PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        由于Transformer不包含递归和卷积，为了让模型利用序列的顺序，必须注入一些关于标记在序列中的相对或绝对位置的信息。
        为此，在编码器和解码器堆栈底部的输入嵌入中添加了“位置编码”，作者使用不同频率的正弦和余弦函数实现。
        :param d_model: (int) embedding后词向量的维度
        :param dropout: (int) 丢弃机制
        :param max_len: (int) 最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 在对数空间计算一次位置编码.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# 一个测试样例
if __name__ == "__main__":
    # 根据位置添加正弦波
    plt.figure(figsize=(15, 5))
    # 设模型维度为20
    pe = PositionalEncoding(20, 0)
    # 执行PE的前向传播，输入张量尺寸为[1, 100, 20]
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # 随便画出几个维度的位置编码
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
    None
