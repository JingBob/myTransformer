from util import *
from decoder import subsequent_mask
from Transformer import make_model
import os


class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        定义一个批处理对象，其中包含用于训练的源句子和目标句子，以及构建掩码。
        :param src: (Tensor)
        :param trg: (Tensor)
        :param pad:
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        生成一个mask来隐藏填充将来出现的词
        :param tgt:
        :param pad:
        :return:
        """
        # 在倒数第二个维度上增加一个维度
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


######################################################################################################################
# @例子1
# 给定来自小词汇表的一组随机输入符号，目标是生成与输入相同的符号，称之为src-tgt copy task
######################################################################################################################
def data_gen(V, batch, nbatches):
    """
    为src-tgt copy task随机生成数据.
    :param V: (int)生成数据的最大值
    :param batch: (int) 批尺寸
    :param nbatches: (int) 批数
    :return: 一个 iterable 对象，
    """
    for i in range(nbatches):
        # 生成[0-V）的随机整数，尺寸为[batch,10]
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # 第一列都置为1
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        # yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，而是返回一个 iterable 对象
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        """
        计算损失的类
        :param generator: 模型的生成器，即transformer最后的预测输出层，对应linea+softmax
        :param criterion: 标签平滑的惩罚项
        :param opt: 优化器
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data.[0] * norm
        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    贪心解码:取解码器输出概率最高的那个词
    :param model:
    :param src:
    :param src_mask:
    :param max_len:
    :param start_symbol:
    :return:
    """
    # memory保存的是编码器输出结果
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == "__main__":
    # 训练src-tgt copy task
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    losslast=float('inf')
    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        # 验证，不需要优化器
        model.eval()
        loss = run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None))
        if loss < losslast:
            if not os.path.exists("model"):
                os.mkdir("model")
            torch.save(model.state_dict(), 'model\copytask.pkl')
            print("save the best model successful!")
        print(loss)

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
