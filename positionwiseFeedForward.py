import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈网络，主要是计算论文的FFN方程：
        除了注意力子层之外，我们的编码器和解码器中的每一层都包含一个完全连接的前馈网络，该网络分别且相同地应用于每个位置。
        这由两个线性变换（全连接层）组成，中间有一个 ReLU 激活。
        :param d_model: (int) 模型维度，论文设置为512
        :param d_ff: (int) 编解码器内层的维度，论文设置为2048
        :param dropout: (int) 丢弃机制，默认0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        # 全连接层nn.Linear实现的就是xW+b
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
