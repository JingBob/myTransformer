import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        Embeddings and Softmax
        与其他序列转导模型类似，使用学习好的嵌入层将输入token和输出token转换为向量.
        使用线性变换和 softmax 函数将decoder输出转换为预测的下一个token概率。
        :param d_model: (int) embedding后词向量维度
        :param vocab: (Tensor)输入词向量
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
