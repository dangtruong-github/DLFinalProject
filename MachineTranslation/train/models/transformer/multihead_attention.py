import torch
import torch.nn as nn

import math


class MultiheadAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = int(d_model / n_head)

        self.q_matrix = nn.Linear(d_model, d_model)
        self.k_matrix = nn.Linear(d_model, d_model)
        self.v_matrix = nn.Linear(d_model, d_model)
        self.o_matrix = nn.Linear(d_model, d_model)

    def split_head(self, x):
        # x: (128, 32, 512)
        batch_size = x.shape[0]

        x = torch.moveaxis(x, 0, 1)
        # (32, 128, 512)

        x = x.reshape(batch_size, -1, self.n_head, self.d_k)
        x = x.permute(0, 2, 1, 3)

        return x
        # (32, 10, 512) => (32, 10, 8, 64) => (32, 8, 10, 64)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, d_model)
        mask
        """

        batch_size = q.shape[0]

        q = self.q_matrix(q)
        k = self.k_matrix(k)
        v = self.v_matrix(v)

        q, k, v = self.split_head(q), self.split_head(k), self.split_head(v)
        # (32, 8, 10, 64)
        k = k.transpose(-2, -1)
        # (32, 8, 64, 10)

        score = torch.matmul(q, k) / math.sqrt(self.d_model)
        # (32, 8, 10, 10)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = nn.functional.softmax(score, dim=-1)

        attn_score = torch.matmul(score, v)
        # (32, 8, 10, 64)

        attn_score = attn_score.transpose(1, 2).contiguous()
        attn_score = attn_score.view(batch_size, -1, self.d_model)

        output = self.o_matrix(attn_score)
        return output
