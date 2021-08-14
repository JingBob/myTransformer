import math
import torch.nn.functional as F
from util import *

"""
@注意力的运用
Transformer 以三种不同的方式使用多头注意力：
1）在“编码器-解码器注意力”层中，查询来自前一个解码器层，记忆键和值来自编码器的输出。
这允许解码器中的每个位置都参与输入序列中的所有位置。模仿了序列到序列模型中典型的编码器-解码器注意机制 。
2）编码器包含自注意力层。在自注意力层中，所有的key、value和query都来自同一个地方，在这种情况下，是编码器上一层的输出。
编码器中的每个位置都可以参与编码器前一层中的所有位置。
3）类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。
我们需要防止解码器中的左向信息流以保留自回归特性。我们通过屏蔽（设置为-∞) softmax 输入中与非法连接相对应的所有value。
"""


def attention(query, key, value, mask=None, dropout=None):
    """
    计算'Scaled Dot Product Attention'，输入由维度为dk的query,key以及维度为dv的values组​​成
    # 注意力函数可以描述为将一个query和一组key对映射到一个output，其中query、keys、values和output都是向量。
    # 输出为values的加权总和，其中分配给每个值的权重由query与相应key的相关性函数计算。
    # 计算key和query的点积，将每个key除以√dk，并应用 softmax 函数来获得值的权重。
    # 在实践中，同时计算一组query的注意力函数，打包成一个矩阵Q
    :param query: (Tensor)query矩阵
    :param key: (Tensor)key矩阵
    :param value: (Tensor)value矩阵
    :param mask:
    :param dropout: (int)丢弃机制
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        计算多头注意力机制，其实就是将'Scaled Dot Product Attention'重复h次
        # 两个最常用的注意力函数是加法注意力 （cite）和点积（乘法）注意力。点积注意力与我们的算法相同
        # 多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。
        :param h: (int)重复次数
        :param d_model: (int) embedding后词向量维度
        :param dropout: (int)丢弃机制
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播，对应论文多头注意力那个图
        :param query: (Tensor)query矩阵
        :param key: (Tensor)key矩阵
        :param value: (Tensor)value矩阵
        :param mask:
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 做线性变换获取query, key, value，模型维度（d_model） 为 h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) 对每个batch所有向量施加注意力.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用view函数和全连接层实现"Concat".
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
