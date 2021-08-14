from util import *
from attention import MultiHeadedAttention
from positionwiseFeedForward import PositionwiseFeedForward
from positional_encoding import PositionalEncoding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from embedding_softmax import Embeddings
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        标准的encoder-decoder架构
        :param encoder: (nn.Module) transformer编码器模型
        :param decoder: (nn.Module) transformer解码器模型
        :param src_embed: (nn.Module) 输入词向量（embedding层）
        :param tgt_embed: (nn.Module) 目标词向量（embedding层）
        :param generator: (nn.Module) 成器，实现transformer的decoder最后的linear+softmax
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        喂入和处理masked src和目标序列.
        decode的输入是encode输出，目标词向量
        :param src: (Tensor) 输入词向量
        :param tgt: (Tensor) 输出词向量
        :param src_mask:
        :param tgt_mask:
        :return: (nn.Module) 整个transformer模型
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """定义标准的linear + softmax生成器."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建整体模型
    :param src_vocab: (int) 输入词向量尺寸
    :param tgt_vocab: (int) 输出词向量尺寸
    :param N: (int, default=6) 编解码层的重复次数
    :param d_model: (int, default=512) embedding后词向量维度
    :param d_ff: (int, default=2048) 编解码器内层维度
    :param h: (int, default=8) 'Scaled Dot Product Attention'，使用的次数
    :param dropout: (int, default=0.1) 丢弃机制，正则化的一种方式，默认为0.1
    :return: (nn.Module) 整个transformer模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 下面这部分非常重要，模型参数使用Xavier初始化方式，基本思想是输入和输出的方差相同，包括前向传播和后向传播
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
