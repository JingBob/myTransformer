from util import *


# 整体Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        搭建整个Transformer的Encoder
        :param layer:(nn.Module)单层Encoder
        :param N: (int)单层Encoder重复的次数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        前向传播
        :param x:
        :param mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        单个Encoder层。Encoder 由 self-attn 和 feed forward 组成 (后面再定义)单个Encoder层。
        :param size: (int)模型尺寸
        :param self_attn: (nn.Module)自注意力层
        :param feed_forward: (nn.Module)前向传播层
        :param dropout: (int)丢弃机制
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向传播，参考论文的连接方式
        :param x: 输入向量
        :param mask:
        :return:
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
