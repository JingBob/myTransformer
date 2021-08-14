import copy
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


def clones(module, N):
    """
    生成N个相同的层
    :param module:(nn.Module)输入模型
    :param N:(int)重复次数
    :return: 复制生成的模型列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        归一化，即每个子层的输出为LayerNorm(x+Sublayer(x)),(x+Sublayer(x)是子层自己实现的功能。
        将 dropout 应用于每个子层的输出，然后再将其添加到子层输入中并进行归一化。
        为了促进这些残差连接，模型中的所有子层以及嵌入层产生维度输出为512
        :param features:
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        """
        残差连接模块，对应论文的 Add & Norm
        :param size: (int)模型尺寸
        :param dropout: (int)丢弃机制
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播，将输入与正则化的输出相加
        :param x:
        :param sublayer:
        :return:
        """
        return x + self.dropout(sublayer(self.norm(x)))


def run_epoch(data_iter, model, loss_compute):
    """
    通用的训练和评分函数来跟踪损失。传入一个通用的损失计算函数处理参数更新。
    :param data_iter:
    :param model:
    :param loss_compute:
    :return:
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        优化器：论文用的是adam，这个类主要用于针对不同模型尺寸动态更新学习率
        :param model_size:
        :param factor:
        :param warmup:
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # 更新参数和学习率
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # 执行上面更新的学习率
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """
    优化器调用示例：
    :param model:
    :return:
    """
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        标签平滑:论文正则化的一种方式，另外就是使用dropout了
        在训练期间，使用values的标签平滑，使用 KL div 损失实现标签平滑，防止模型过度自信预测
        论文没有使用 one-hot 目标分布，而是创建了一个分布，该分布具有confidence正确的单词和分布在整个词汇表中的其余smoothing。
        :param size: (int) 模型尺寸，对应词向量长度
        :param padding_idx: (int) 填充步幅
        :param smoothing:
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == "__main__":
    # 针对不同模型大小和优化超参数的曲线示例。
    # 设置三个不同的模型尺寸，最大学习率上升步阈值
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()

    # 测试 label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
    print(predict.log())
    print(crit.true_dist)
    # 显示系统预期的目标分布.
    plt.imshow(crit.true_dist)
    plt.show()

    # 标签平滑实际上开始惩罚模型，如果它对给定的选择非常自信。
    crit = LabelSmoothing(5, 0, 0.1)
    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
        print("predict:", predict)
        return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()
