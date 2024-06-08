import torch
import torch.nn as nn

import math


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pos_even_use = pos / (10000 ** ((2 * i) / self.d_model))
                pe[pos, i] = math.sin(pos_even_use)

                pos_odd_use = pos / (10000 ** ((2 * (i + 1)) / self.d_model))
                pe[pos, i + 1] = math.cos(pos_odd_use)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        seq_len = x.size(1)
        x += torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x
