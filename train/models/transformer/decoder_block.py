import torch.nn as nn

from train.models.transformer.multihead_attention import MultiheadAttention
from train.models.transformer.transformer_block import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, factor=4, n_head=8):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.n_head = n_head

        self.attention = MultiheadAttention(d_model, n_head=8)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(d_model, factor=factor,
                                                  n_head=n_head)

    def forward(self, x, k, v, mask):
        attention = self.attention(x, x, x, mask=mask)
        q = self.dropout(self.norm(attention + x))
        out = self.transformer_block(q, k, v)

        return out
