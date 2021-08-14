from util import *


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        搭建整个Transformer的Decoder
        :param layer: (nn.Module)单层Encoder
        :param N: (int)单层Encoder重复的次数
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        解码器由 self-attn, src-attn 和 feed forward组成
        除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意。
        与编码器类似，在每个子层周围使用残差连接，然后进行层归一化。
        :param size:
        :param self_attn:
        :param src_attn:
        :param feed_forward:
        :param dropout:
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    用于遮住序列的一些位置
    修改了解码器中的自注意力子层，以防止位置关注后续位置。
    这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测只能依赖小于位置i的已知输出
    :param size: (int)向量长度
    :return: (Tensor,bool)掩码后的矩阵，尺寸为[1,size,size]
    """
    attn_shape = (1, size, size)
    # 返回函数的上三角矩阵，从k=1列开始
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


if __name__ == "__main__":
    # 注意mask下方显示了每个tgt单词（行）允许查看的位置（列）。单词被屏蔽，以便在训练期间注意将来的单词。
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()
